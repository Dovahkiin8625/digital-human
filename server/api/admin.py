import json
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from starlette.responses import FileResponse, Response

from server.api.avatars import _create_avatar_rows, _derive_suffix, _require_uuid_hex
from server.avatar import liveportrait  # noqa: F401 -- side-effect: registers the liveportrait backend
from server.avatar import wan22  # noqa: F401 -- side-effect: registers the two Wan 2.2 I2V variants
from server.avatar import musetalk_silent  # noqa: F401 -- side-effect: registers the musetalk-silent backend
from server.avatar.pipeline import run_job_in_new_session
from server.avatar import video
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob, StreamSession
from server.db.session import get_db


def _artifact_kind(path: Path) -> str:
    """Categorise the file so the UI can pick an icon/colour."""
    name = path.name.lower()
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".pt") or name.endswith(".pth") or name.endswith(".pkl") or name.endswith(".safetensors"):
        return "tensor"
    if name.endswith(".mp4") or name.endswith(".mov") or name.endswith(".webm") or name.endswith(".mkv"):
        return "video"
    # The biSeNet masks and the source video keep the original extension.
    return "binary"


def _list_avatar_artifacts(avatar: Avatar) -> list[dict]:
    """Walk ``<data_dir>/avatars/<id>/`` and return a stable UI shape.

    Empty list if the avatar has never been prepared (no avatar dir yet). The
    UI uses this to render a "Built artifacts" list with download buttons.
    """
    store = AvatarStore(get_settings().data_dir)
    root = store.avatar_dir(avatar.id)
    if not root.exists():
        return []
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == root / "pack.json":
            # pack.json is exposed via the config block already; skip duplication.
            continue
        rel = path.relative_to(root).as_posix()
        rows.append({"path": rel, "size": path.stat().st_size, "kind": _artifact_kind(path)})
    if (root / "pack.json").exists():
        rows.append(
            {
                "path": "pack.json",
                "size": (root / "pack.json").stat().st_size,
                "kind": "json",
            }
        )
    return rows


def _pipeline_timeline(avatar: Avatar, raw_meta: dict | None) -> dict:
    """Compose the admin-side timeline view from ``Avatar.config_json``.

    Mock and MuseTalk both write a config_json; both have ``prep.mode``. The
    MuseTalk pack adds ``crop``, ``detector``, ``version``, ``fps``,
    ``source_fps``, ``frame_count``, plus an optional ``warnings`` list.

    For mock avatars (no ``prep`` block, no ``musetalk/meta.json``) we fall
    back to the source file's suffix so the UI still gets a usable mode.
    """
    cfg = dict(avatar.config_json or {})
    prep = cfg.get("musetalk", {}) if isinstance(cfg.get("musetalk"), dict) else {}
    meta = dict(raw_meta or {})
    mode = prep.get("mode") or meta.get("mode")
    if not mode:
        suffix = Path(avatar.source_image_path).suffix.lower()
        mode = "video" if video.is_video_suffix(suffix) else "image"
    out: dict = {
        "mode": mode,
        "fps": prep.get("fps") or meta.get("fps"),
        "source_fps": prep.get("source_fps") or meta.get("source_fps"),
        "frame_count": prep.get("frame_count") or meta.get("frame_count"),
        "crop": prep.get("crop") or meta.get("crop"),
        "detector": prep.get("detector") or meta.get("detector"),
        "version": prep.get("version") or meta.get("version"),
        "warnings": cfg.get("warnings") or meta.get("warnings") or [],
    }
    kept = meta.get("kept")
    if isinstance(kept, list):
        out["kept_count"] = len(kept)
    return out

# Single router: no auth dependencies, all endpoints public. ``admin-ui``
# serves the same console regardless of who hits the API; the box is
# assumed to be a single-operator machine (per the no-tenant auth/tenant
# refactor).
router = APIRouter(prefix="/api/admin", tags=["admin"])


class TtsIn(BaseModel):
    text: str
    voice_id: str | None = None


@router.get("/system")
def system_info() -> dict:
    from server.main import get_session_manager

    # The shared engine, not a fresh one per request: a real engine's describe()
    # is cheap, but constructing one per request would be a new thread pool and
    # a new view of the model state every poll (the page polls every 2 s).
    described = get_session_manager().engine.describe()
    settings = get_settings()
    return {
        "engine": described,
        "pipeline": settings.avatar_pipeline,
        "version": "0.1.0",
        "data_dir": str(settings.data_dir),
        "gpu": described.get("gpu", False),
        "frame_width": settings.frame_width,
        "frame_height": settings.frame_height,
        "fps": settings.fps,
        "sample_rate": settings.sample_rate,
        "video_bitrate": settings.video_bitrate,
        "video_max_bitrate": settings.video_max_bitrate,
        "session_idle_timeout": settings.session_idle_timeout,
        "session_sweep_interval": settings.session_sweep_interval,
        "avatar_max_video_seconds": settings.avatar_max_video_seconds,
        "avatar_max_upload_bytes": settings.avatar_max_upload_bytes,
        "minimax_voice_id": settings.minimax_voice_id,
        "minimax_sample_rate": settings.minimax_sample_rate,
    }


@router.get("/settings")
def admin_get_settings(db=Depends(get_db)) -> dict:
    """Effective tunables as a flat dict; defaults overridden by stored keys."""
    from server.avatar.tunables import effective

    return effective(db)


@router.patch("/settings")
def admin_patch_settings(payload: dict, db=Depends(get_db)) -> dict:
    """Validate, persist overrides, return the full effective dict.

    Partial updates are fine: any number of keys can come in. Unknown keys or
    out-of-range values are rejected as a single ``error.message`` so the UI can
    point at the bad field.
    """
    from datetime import datetime, timezone

    from server.avatar.tunables import defaults, dump_json, validate
    from server.db.models import Setting

    errors = validate(payload)
    if errors:
        raise AppError("invalid_settings", "one or more settings are invalid", 400, detail=errors)
    for name, value in payload.items():
        row = db.get(Setting, name)
        if row is None:
            row = Setting(key=name, value=dump_json(value))
            db.add(row)
        else:
            row.value = dump_json(value)
            row.updated_at = datetime.now(timezone.utc)
    db.commit()
    # Re-read defaults + overrides for a single coherent response.
    from server.avatar.tunables import effective

    return effective(db)


@router.post("/i2v", status_code=202)
def admin_generate_i2v(
    background: BackgroundTasks,
    backend: str = Form(...),
    file: UploadFile = ...,
    prompt: str = Form(""),
    num_frames: int = Form(...),
    steps: int = Form(0),
    driver: str = Form("subtle"),
    db=Depends(get_db),
) -> dict:
    """Submit an image-to-video generation as an async task.

    The admin UI's "图生视频" tab calls this. The endpoint does not block on
    the encoder (which can run for tens of seconds): it validates the
    request synchronously, persists a ``GenerationTask`` row, and schedules
    the actual run on a FastAPI background task. Returns ``202`` with
    ``task_id``; the operator polls ``/api/admin/tasks/{id}`` until the
    row reaches a terminal state, then downloads the mp4 from
    ``/api/admin/tasks/{id}/result.mp4``.

    ``backend`` selects one of the portrait animators in
    ``server/avatar/animators.ANIMATORS``; per-call knobs
    (``prompt`` / ``num_frames`` / ``steps`` / ``driver``) are dispatched
    into the chosen backend's own handler so each one sees the constraints
    it understands. ``driver`` is LivePortrait-specific -- other backends
    ignore it. Synchronous validation (unknown backend, out-of-range
    knob) still surfaces as 400 -- a doomed task should never enter the
    table.
    """
    from server.avatar.animators import ANIMATORS, AnimatorError
    from server.db.models import GenerationTask

    if backend not in ANIMATORS:
        # Surface the *known* names so the operator (or the page's stale
        # bundle) can pick a valid one without trial-and-error.
        valid = ", ".join(sorted(ANIMATORS)) or "(none registered)"
        raise AppError(
            "invalid_backend",
            f"unknown backend {backend!r}; choose one of {valid}",
            400,
            detail={"backend": f"choose one of: {valid}"},
        )
    animator = ANIMATORS[backend]
    # The backend's own validator decides which knobs are honoured. Knobs
    # it doesn't understand are dropped silently rather than 422'd -- the
    # backend knows its own contract and tells the page so via its metadata.
    field_errors = animator.validate(
        prompt=str(prompt or ""), num_frames=int(num_frames), steps=int(steps)
    )
    if field_errors:
        raise AppError("invalid_animator_params", "animator params out of range", 400, detail=field_errors)

    # Persist the upload into a per-task scratch dir first; the background
    # worker reads from disk, not from the (closed) request body. The dir
    # lives under ``tmp/anim/<task_id>`` so a later re-render of the same
    # task (or a sweep) can find its inputs by id.
    task = GenerationTask(
        type="i2v",
        status="queued",
        stage="",
        progress=0,
        params_json=json.dumps(
            {
                "backend": backend,
                "prompt": str(prompt or ""),
                "num_frames": int(num_frames),
                "steps": int(steps),
                "driver": str(driver or "subtle"),
                "filename": file.filename,
            }
        ),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    scratch_root = Path(get_settings().data_dir) / "tmp" / "anim"
    scratch_root.mkdir(parents=True, exist_ok=True)
    subdir = scratch_root / task.id
    subdir.mkdir(parents=True, exist_ok=True)
    suffix = _derive_suffix(file.filename)
    source_path = subdir / f"src{suffix}"
    source_path.write_bytes(file.file.read())

    # Schedule the real run. ``run_i2v_task`` owns its own DB session so the
    # caller's session can close immediately without breaking the work.
    background.add_task(_run_i2v_task, task.id, subdir, source_path)
    return {"task_id": task.id}


def _run_i2v_task(task_id: str, subdir: Path, source_path: Path) -> None:
    """Background runner for an i2v GenerationTask.

    Owns its own DB session (the submitter's session is gone by the time we
    reach here), owns its scratch dir cleanup, and reports every state
    transition back to the row so the operator's poll picks it up.

    Any exception inside the animator is captured into the row's ``error``
    column -- the task transitions to ``failed`` and the HTTP client never
    sees a 5xx for problems it could not prevent.
    """
    import json as _json
    from datetime import datetime, timezone

    from sqlalchemy.orm import sessionmaker

    from server.avatar.animators import ANIMATORS, AnimatorError
    from server.config import get_settings
    from server.db.models import GenerationTask
    from server.db.session import get_engine

    session = sessionmaker(bind=get_engine())()
    try:
        task = session.get(GenerationTask, task_id)
        if task is None:
            # The submitter deleted the task between submit and run; nothing
            # we can do, just drop the scratch.
            shutil.rmtree(subdir, ignore_errors=True)
            return

        task.status = "running"
        task.stage = "encoding"
        task.progress = 10
        task.updated_at = datetime.now(timezone.utc)
        session.commit()

        params = _json.loads(task.params_json)
        animator = ANIMATORS[params["backend"]]

        # ``driver`` is a LivePortrait-specific knob. Other backends
        # (musetalk_silent, wan22) don't accept it -- only forward it when
        # the chosen backend speaks it, so the dispatcher stays type-safe
        # without each backend having to declare ``**kwargs``.
        import inspect

        sig = inspect.signature(animator.animate)
        kwargs = dict(
            prompt=params["prompt"],
            num_frames=params["num_frames"],
            steps=params["steps"],
            out_dir=subdir,
        )
        if "driver" in sig.parameters:
            kwargs["driver"] = params["driver"]

        try:
            animator.animate(source_path, **kwargs)
        except AnimatorError as exc:
            task.status = "failed"
            task.error = f"animator: {exc}"[:1024]
            task.stage = ""
            task.updated_at = datetime.now(timezone.utc)
            session.commit()
            return
        except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
            task.status = "failed"
            task.error = str(exc)[:1024]
            task.stage = ""
            task.updated_at = datetime.now(timezone.utc)
            session.commit()
            return

        clip = subdir / "out.mp4"
        if not clip.exists():
            task.status = "failed"
            task.error = "animator produced no clip"
            task.stage = ""
            task.updated_at = datetime.now(timezone.utc)
            session.commit()
            return

        task.status = "done"
        task.stage = "done"
        task.progress = 100
        task.result_json = _json.dumps({"mp4_path": str(clip)})
        task.updated_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()


def _serialize_task(task: "GenerationTask") -> dict:
    """Render a GenerationTask as the API shape the UI consumes."""
    import json as _json

    out: dict = {
        "task_id": task.id,
        "type": task.type,
        "status": task.status,
        "stage": task.stage,
        "progress": task.progress,
        "params": _json.loads(task.params_json) if task.params_json else {},
        "result": _json.loads(task.result_json) if task.result_json else None,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }
    return out


@router.get("/tasks")
def admin_list_tasks(db=Depends(get_db)) -> dict:
    """List recent generation tasks newest-first.

    Used by the "任务" tab to populate its table. The list is bounded so a
    long-running server doesn't return a multi-thousand-row payload; active
    tasks are pinned to the top so the operator always sees what's running
    first.
    """
    from server.db.models import GenerationTask

    rows = (
        db.execute(select(GenerationTask).order_by(GenerationTask.created_at.desc()).limit(200))
        .scalars()
        .all()
    )
    return {"tasks": [_serialize_task(r) for r in rows]}


@router.get("/tasks/{task_id}")
def admin_get_task(task_id: str, db=Depends(get_db)) -> dict:
    """One task by id. Used by the i2v page's progress poll."""
    from server.db.models import GenerationTask

    _require_uuid_hex(task_id)
    row = db.get(GenerationTask, task_id)
    if row is None:
        raise AppError("not_found", f"task {task_id} not found", 404)
    return _serialize_task(row)


@router.delete("/tasks/{task_id}", status_code=204)
def admin_delete_task(task_id: str, db=Depends(get_db)) -> Response:
    """Remove a generation task's DB row + on-disk scratch dir.

    Wired to the 图生视频 page's per-row 删除 button. The on-disk scratch
    lives at ``<data_dir>/tmp/anim/<task_id>`` (set by ``admin_generate_i2v``);
    failing to clean it would leave dead ends under ``tmp/anim`` over the
    operator's career. The background runner already guards on the row
    being gone (``_run_i2v_task`` short-circuits if ``db.get(task_id)`` is
    None), so deleting a still-running task is safe -- the runner just
    no-ops.
    """
    from server.db.models import GenerationTask

    _require_uuid_hex(task_id)
    row = db.get(GenerationTask, task_id)
    if row is None:
        raise AppError("not_found", f"task {task_id} not found", 404)
    scratch = Path(get_settings().data_dir) / "tmp" / "anim" / task_id
    shutil.rmtree(scratch, ignore_errors=True)
    db.delete(row)
    db.commit()
    return Response(status_code=204)


@router.get("/tasks/{task_id}/result.mp4")
def admin_get_task_result(task_id: str, db=Depends(get_db)) -> Response:
    """Stream the mp4 bytes that the i2v animator wrote.

    Only meaningful for tasks with ``status == "done"``; a still-running or
    failed task returns 404 so the page doesn't accidentally try to play
    nothing. The bytes are read from the on-disk path the background runner
    recorded in ``result_json``.
    """
    from server.db.models import GenerationTask

    _require_uuid_hex(task_id)
    row = db.get(GenerationTask, task_id)
    if row is None:
        raise AppError("not_found", f"task {task_id} not found", 404)
    if row.status != "done" or not row.result_json:
        raise AppError("not_ready", f"task {task_id} is not ready ({row.status})", 404)
    payload = json.loads(row.result_json)
    path = Path(payload["mp4_path"])
    if not path.exists():
        # The scratch dir was swept out from under us -- surface that as a
        # 410 so the page knows the result isn't recoverable, not a 500.
        raise AppError("result_missing", f"task {task_id} result is gone", 410)
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="out.mp4"'},
    )


# ----- LivePortrait drivers -------------------------------------------------

@router.get("/liveportrait-drivers")
def admin_list_liveportrait_drivers() -> dict:
    """Return the list of LivePortrait driving clips available on disk.

    Each entry is one driver stem with the metadata the ``形象活化`` tab
    needs to render its dropdown:

    * ``name`` -- the driver id; the UI/API speak this.
    * ``filename`` -- the actual file the resolver will pick.
    * ``kind`` -- ``"pkl"`` (pre-extracted motion template, fastest) or
      ``"mp4"`` (raw video; upstream will crop + extract on first run).
    * ``cached`` -- ``True`` iff a ``.pkl`` exists alongside the ``.mp4``.
      An ``"mp4"``-kind driver with ``cached=True`` will be promoted to
      ``"pkl"`` automatically on the next ``animate()`` call, and the
      endpoint re-reads the directory each request so the flag flips
      immediately after the first run.
    * ``canonical`` -- ``True`` for the names populated by
      ``scripts/fetch_liveportrait_weights.sh``; user-uploaded extras
      surface here too so the UI doesn't have to fork the registry.
    """
    from server.avatar import liveportrait

    return {"drivers": liveportrait.list_drivers()}


def _validate_template_name(name: str) -> None:
    """Reject names that would break the on-disk layout.

    The name becomes a file stem under :data:`LIVE_PORTRAIT_DRIVERS_DIR`,
    so it must be a legal filename across the platforms we ship to
    (Windows + Linux). Letters (ASCII + CJK), digits, underscores, hyphens,
    and dots-in-the-middle are allowed; leading dots and Windows-reserved
    characters are not.
    """
    if not name or name in (".", ".."):
        raise AppError(
            "invalid_driver_name",
            f"template name {name!r} is reserved",
            400,
        )
    # Block characters Windows / POSIX both reject in filenames, plus the
    # ASCII control range. CJK is fine (NTFS, ext4, APFS all support it).
    forbidden = set('\\/:*?"<>|\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09'
                    '\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15'
                    '\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
    if any(ch in forbidden for ch in name):
        bad = "".join(sorted(forbidden & set(name)))
        raise AppError(
            "invalid_driver_name",
            f"template name {name!r} contains an illegal character "
            f"(one of: {bad!r})",
            400,
        )
    if name.startswith("."):
        raise AppError(
            "invalid_driver_name",
            f"template name {name!r} may not start with a dot (hidden file)",
            400,
        )
    if len(name) > 120:
        raise AppError(
            "invalid_driver_name",
            f"template name is too long ({len(name)} chars; max 120)",
            400,
        )


def _ffmpeg_trim(src: Path, dst: Path, max_seconds: float) -> None:
    """Run ``ffmpeg -t`` to clip ``src`` to ``max_seconds`` seconds.

    Uses stream copy when possible (no re-encode) so a 30 s upload becomes
    a 5 s clip in <100 ms. ``ffmpeg`` is on PATH (the project already
    requires it for the mp4 muxer; this reuses the same install).
    """
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(src),
            "-t", f"{max_seconds:.3f}",
            "-c", "copy",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def _ffmpeg_gif(src: Path, dst: Path, max_seconds: float = 5.0) -> None:
    """Render a small, palette-optimised GIF preview from ``src``.

    240 px wide, 12 fps, capped at ``max_seconds`` (5 by default) -- the
    preview is meant to show what motion the template carries, not to
    play the full clip. The two-pass palette (``palettegen`` +
    ``paletteuse``) keeps file size sane (~200 KB for a 5 s preview).
    """
    import subprocess

    # First pass: derive a global palette from the source.
    palette = src.with_suffix(".palette.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(src),
            "-t", f"{max_seconds:.3f}",
            "-vf", "fps=12,scale=240:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    try:
        # Second pass: scale + apply palette.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(src),
                "-t", f"{max_seconds:.3f}",
                "-i", str(palette),
                "-filter_complex",
                "fps=12,scale=240:-1:flags=lanczos[x];[x][1:v]paletteuse",
                "-loop", "0",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        palette.unlink(missing_ok=True)


@router.post("/liveportrait-drivers/extract")
def admin_extract_liveportrait_driver_template(
    name: str = Form(...),
    video: UploadFile = ...,
    max_seconds: float = Form(0.0),
) -> dict:
    """Upload a driving video and pre-extract its motion template.

    Writes ``<name>.mp4`` to :data:`LIVE_PORTRAIT_DRIVERS_DIR`, runs the
    upstream pipeline against a vendored placeholder source image (so the
    crop + feature extraction happens), and keeps the resulting
    ``<name>.pkl`` next to the mp4. The generated mp4 output is discarded;
    only the template is kept.

    Optional ``max_seconds`` clips the upload before extraction so the
    template's frame count matches the operator's intended duration
    rather than the full upload length. The clip is done with
    ``ffmpeg -c copy`` (no re-encode) so a 30 s upload becomes a 5 s clip
    in milliseconds.

    On success, a small palette-optimised ``<name>.gif`` preview is also
    written so the system menu and the 形象活化 page can show the
    motion at a glance.

    Returns the new template's metadata. Errors out if ``name`` collides
    with an existing driver file, or if the extraction pipeline can't
    process the upload (no face in the source, weights missing, ...).
    """
    from server.avatar import liveportrait

    _validate_template_name(name)
    if max_seconds < 0:
        raise AppError(
            "invalid_max_seconds",
            f"max_seconds must be >= 0 (0 = use full video)",
            400,
        )
    drivers_dir = liveportrait.LIVE_PORTRAIT_DRIVERS_DIR
    drivers_dir.mkdir(parents=True, exist_ok=True)
    target_mp4 = drivers_dir / f"{name}.mp4"
    if target_mp4.exists():
        raise AppError(
            "driver_exists",
            f"driver {name!r} already exists at {target_mp4}; delete it first",
            409,
        )

    # Persist the upload first so we can probe/trim it.
    target_mp4.write_bytes(video.file.read())
    driving_for_upstream = target_mp4
    try:
        if max_seconds > 0:
            # Clip to the first ``max_seconds`` -- a fast stream-copy re-mux
            # so we don't pay for a re-encode on what is potentially a
            # 30 s source video.
            clipped = target_mp4.with_suffix(".clip.mp4")
            try:
                _ffmpeg_trim(target_mp4, clipped, max_seconds)
                driving_for_upstream = clipped
            except Exception as exc:  # noqa: BLE001 -- ffmpeg missing or unhappy
                raise AppError(
                    "liveportrait_trim_failed",
                    f"failed to trim the upload to {max_seconds:g}s: {exc}. "
                    f"is ffmpeg on PATH?",
                    400,
                )
    except AppError:
        # Clean up the persisted upload on early failure so a follow-up
        # retry with the same name doesn't hit the 409 above.
        target_mp4.unlink(missing_ok=True)
        raise

    try:
        from server.avatar.liveportrait import LivePortraitAnimator  # noqa: F401 -- side-effect import
    except Exception as exc:  # noqa: BLE001 -- setup path
        target_mp4.unlink(missing_ok=True)
        raise AppError(
            "liveportrait_unavailable",
            f"LivePortrait runtime not installed ({exc}); "
            f"run scripts/setup_liveportrait.sh first",
            503,
        )

    from server.config import get_settings

    settings = get_settings()
    # Vendored placeholder source. Upstream's ``make_abs_path`` joins
    # against its config dir, so passing an absolute path here bypasses
    # that and gives us a stable, version-controlled face for the
    # extraction pass.
    placeholder_src = (
        Path(settings.data_dir).parent  # project root, since data_dir is under it
        / "vendor"
        / "LivePortrait-upstream"
        / "assets"
        / "examples"
        / "source"
        / "s0.jpg"
    )
    if not placeholder_src.exists():
        target_mp4.unlink(missing_ok=True)
        raise AppError(
            "liveportrait_no_placeholder",
            f"placeholder source missing at {placeholder_src}; "
            f"is the LivePortrait upstream checkout intact?",
            503,
        )

    scratch_root = Path(settings.data_dir) / "tmp" / "liveportrait_extract"
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=scratch_root))
    target_pkl = drivers_dir / f"{name}.pkl"
    try:
        try:
            from LivePortrait.live_portrait_pipeline import LivePortraitPipeline  # type: ignore[import-not-found]
            from src.config.argument_config import ArgumentConfig  # type: ignore[import-not-found]
            from src.config.inference_config import InferenceConfig  # type: ignore[import-not-found]
            from src.config.crop_config import CropConfig  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AppError(
                "liveportrait_unavailable",
                f"LivePortrait runtime not installed ({exc}); "
                f"run scripts/setup_liveportrait.sh first",
                503,
            )

        weights = Path(settings.data_dir) / "models" / "liveportrait"
        inference_cfg = InferenceConfig(
            checkpoint_F=str(weights / "liveportrait" / "base_models" / "appearance_feature_extractor.pth"),
            checkpoint_M=str(weights / "liveportrait" / "base_models" / "motion_extractor.pth"),
            checkpoint_G=str(weights / "liveportrait" / "base_models" / "spade_generator.pth"),
            checkpoint_W=str(weights / "liveportrait" / "base_models" / "warping_module.pth"),
            checkpoint_S=str(weights / "liveportrait" / "retargeting_models" / "stitching_retargeting_module.pth"),
        )
        crop_cfg = CropConfig(
            insightface_root=str(weights / "insightface"),
            landmark_ckpt_path=str(weights / "liveportrait" / "landmark.onnx"),
        )
        pipe = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        args = ArgumentConfig(
            source=str(placeholder_src.absolute()),
            driving=str(driving_for_upstream.absolute()),
            output_dir=str(scratch.absolute()),
        )
        try:
            pipe.execute(args)
        except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
            raise AppError(
                "liveportrait_extract_failed",
                f"LivePortrait extraction failed: {exc}",
                502,
            )

        # Upstream dumped ``<name>.pkl`` next to the driving file (its
        # ``wfp_template`` resolution). Read its size for the response.
        if not target_pkl.exists():
            raise AppError(
                "liveportrait_extract_no_template",
                f"extraction completed but no template at {target_pkl}; "
                f"check upstream logs",
                502,
            )
        size = target_pkl.stat().st_size
        # Read frame count from the template dict.
        import pickle

        with open(target_pkl, "rb") as f:
            n_frames = int(pickle.load(f).get("n_frames", 0))
        # Cache the frame count as a sidecar so the API can surface it
        # without re-reading the pkl every listing. The fetch script does
        # the same for the bundled drivers.
        liveportrait.write_frame_count_sidecar(name, n_frames)

        # Render a short GIF preview from the (clipped) driving video. A
        # best-effort pass -- if ffmpeg is missing the operator can re-run
        # later, but extraction itself has already succeeded.
        target_gif = drivers_dir / f"{name}.gif"
        gif_size = 0
        try:
            # For pkl-only drivers we'd render from the mp4 (it's the
            # same visual content). For very-short templates the GIF ends
            # up small.
            _ffmpeg_gif(driving_for_upstream, target_gif, max_seconds=5.0)
            gif_size = target_gif.stat().st_size if target_gif.exists() else 0
        except Exception as exc:  # noqa: BLE001 -- preview is non-critical
            # Leave the .gif missing; ``list_drivers`` reports has_preview
            # accordingly. Don't fail the whole request.
            target_gif.unlink(missing_ok=True)
            gif_size = 0
            _gif_warn = exc  # noqa: F841 -- retained only for the trace below

        return {
            "name": name,
            "filename": target_pkl.name,
            "size_bytes": size,
            "frames": n_frames,
            "source_mp4": target_mp4.name,
            "preview_gif": target_gif.name if gif_size else None,
            "preview_bytes": gif_size,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        # Drop the temp clip; the original upload is now at ``target_mp4``
        # (and is what ``list_drivers`` will report as ``source_mp4``).
        clipped_path = target_mp4.with_suffix(".clip.mp4")
        clipped_path.unlink(missing_ok=True)


@router.get("/liveportrait-drivers/{name}/preview")
def admin_get_liveportrait_driver_preview(name: str) -> Response:
    """Serve the rendered GIF preview for a motion template.

    Intentionally **unauthenticated**: the browser's ``<img>`` tag can't
    send custom ``X-Admin-Token`` headers, and a pre-rendered GIF is no
    more sensitive than the listing endpoint that already reveals the
    same driver names + sizes to anyone with the admin token. The
    ``listing`` endpoint stays behind auth; this one serves the bytes.

    404 when the template has no ``.gif`` yet (e.g. bundled drivers on
    a host where the fetch script ran before the GIF generation was
    added; the operator can re-run ``scripts/fetch_liveportrait_weights.sh``
    to backfill them).
    """
    from server.avatar import liveportrait

    drivers_dir = liveportrait.LIVE_PORTRAIT_DRIVERS_DIR
    # Path-traversal guard: only accept the same name shape the create
    # endpoint validates. ``name`` came from the URL path; without this
    # check a ``..`` could escape ``drivers_dir``.
    if "/" in name or "\\" in name or name in (".", ".."):
        raise AppError("no_preview", f"invalid template name {name!r}", 404)
    gif_path = drivers_dir / f"{name}.gif"
    if not gif_path.is_file():
        raise AppError(
            "no_preview",
            f"no preview for {name!r}; the operator can re-run "
            f"scripts/fetch_liveportrait_weights.sh or re-extract the template",
            404,
        )
    return Response(
        content=gif_path.read_bytes(),
        media_type="image/gif",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.delete("/liveportrait-drivers/{name}", status_code=204)
def admin_delete_liveportrait_driver(name: str) -> Response:
    """Remove a driver's mp4 and/or pkl from disk.

    204 on success regardless of which files were present. Refuses to touch
    drivers in :data:`_DRIVER_SOURCES` -- those are populated by the
    fetch script and the operator should re-run that script to restore
    them, not edit them piecemeal here.
    """
    from server.avatar import liveportrait

    if name in liveportrait._DRIVER_SOURCES:
        raise AppError(
            "driver_protected",
            f"driver {name!r} is part of the bundled set; "
            f"re-run scripts/fetch_liveportrait_weights.sh to restore it",
            400,
        )
    drivers_dir = liveportrait.LIVE_PORTRAIT_DRIVERS_DIR
    for ext in (".mp4", ".pkl", ".gif", ".frames"):
        path = drivers_dir / f"{name}{ext}"
        path.unlink(missing_ok=True)
    return Response(status_code=204)


@router.get("/backends")
def admin_list_backends() -> dict:
    """Surface the *loaded* portrait animators to the "形象活化" tab.

    Each backend declares its own ``meta()`` block (label, VRAM hint, which
    knobs it honours); the page renders them in a grouped dropdown. The
    listing is filtered to ``is_loaded=True`` so the dropdown only shows
    options the operator can actually click without getting a 503 --
    the dispatcher's animate() refuses any backend whose pipeline hasn't
    been pre-allocated, and listing a not-loaded backend in the dropdown
    would set the operator up for a guaranteed-failure click.

    An empty registry is a valid state -- the page renders a placeholder
    card and no backend is selectable. The page also re-fetches this
    listing as the operator loads / unloads entries on the Models tab,
    so a freshly-loaded LivePortrait row appears here within one poll.
    """
    from server.avatar.animators import ANIMATORS

    out: list[dict] = []
    for name, animator in ANIMATORS.items():
        try:
            loaded = bool(getattr(animator, "is_loaded", False))
        except Exception:  # noqa: BLE001 -- the property may probe torch
            loaded = False
        if not loaded:
            continue
        meta = animator.meta()
        out.append({"name": name, **meta})
    return {"backends": out}


@router.get("/avatars")
def admin_list_avatars(db=Depends(get_db)) -> list[dict]:
    rows = db.execute(select(Avatar).order_by(Avatar.created_at.desc())).scalars().all()
    return [_admin_avatar_row(db, a) for a in rows]


def _admin_avatar_row(db, a: Avatar) -> dict:
    """Full admin shape for an avatar, matching the list rows (incl. latest job error)."""
    job = (
        db.execute(
            select(AvatarJob)
            .where(AvatarJob.avatar_id == a.id)
            .order_by(AvatarJob.created_at.desc())
        )
        .scalars()
        .first()
    )
    return {
        "avatar_id": a.id,
        "name": a.name,
        "status": a.status,
        "config": a.config_json,
        "error": job.error if a.status == "failed" and job else None,
        "created_at": a.created_at.isoformat(),
    }


@router.post("/avatars", status_code=201)
def admin_create_avatar(
    background: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = ...,
    idle_crf: str = Form(""),
    idle_preset: str = Form(""),
    db=Depends(get_db),
) -> dict:
    suffix = _derive_suffix(file.filename)
    # Cheap early refusal when the client declared an oversized body. This is
    # only an optimization: the counted copy in save_source is authoritative
    # (declared length can lie), and works when size is unknown.
    size = getattr(file, "size", None)
    if size is not None and int(size) > get_settings().avatar_max_upload_bytes:
        raise AppError(
            "file_too_large",
            f"upload is larger than {get_settings().avatar_max_upload_bytes} bytes",
            413,
        )

    # Capture per-avatar tunable overrides (idle_crf / idle_preset) from the
    # create form. Empty fields mean "no override -- use the global value".
    # We validate eagerly so a bad form is a 400 before any row gets written.
    overrides: dict = {}
    field_errors: dict[str, str] = {}
    if idle_crf != "":
        from server.avatar.tunables import validate

        try:
            crf_val = int(idle_crf)
        except ValueError:
            field_errors["idle_crf"] = "must be an integer"
        else:
            errs = validate({"idle_crf": crf_val})
            if errs:
                field_errors["idle_crf"] = errs["idle_crf"]
            else:
                overrides["idle_crf"] = crf_val
    if idle_preset != "":
        from server.avatar.tunables import validate

        errs = validate({"idle_preset": idle_preset})
        if errs:
            field_errors["idle_preset"] = errs["idle_preset"]
        else:
            overrides["idle_preset"] = idle_preset
    if field_errors:
        raise AppError(
            "invalid_avatar_overrides",
            "one or more per-avatar overrides are invalid",
            400,
            detail=field_errors,
        )

    avatar_id, job_id = _create_avatar_rows(
        db, name, file.file, suffix, overrides or None
    )
    background.add_task(run_job_in_new_session, job_id, AvatarStore(get_settings().data_dir))
    return {"avatar_id": avatar_id, "status": "pending"}


@router.get("/avatars/{avatar_id}")
def admin_get_avatar(avatar_id: str, db=Depends(get_db)) -> dict:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return _admin_avatar_row(db, avatar)


@router.delete("/avatars/{avatar_id}", status_code=204)
def admin_delete_avatar(avatar_id: str, db=Depends(get_db)) -> Response:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    db.execute(sa_delete(AvatarJob).where(AvatarJob.avatar_id == avatar_id))
    db.delete(avatar)
    db.commit()
    AvatarStore(get_settings().data_dir).delete_assets(avatar_id)
    return Response(status_code=204)


@router.post("/avatars/{avatar_id}/retry", status_code=202)
def retry_avatar(avatar_id: str, background: BackgroundTasks, db=Depends(get_db)) -> dict:
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    job = AvatarJob(id=uuid4().hex, avatar_id=avatar_id, stage="queued")
    db.add(job)
    avatar.status = "pending"
    db.commit()
    background.add_task(run_job_in_new_session, job.id, AvatarStore(get_settings().data_dir))
    return {"job_id": job.id}


@router.get("/avatars/{avatar_id}/idle.mp4")
def avatar_idle_video(avatar_id: str, db=Depends(get_db)):
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    if pack.prep.get("mode") != "video":
        # Photo/legacy avatars have no source cycle; the UI uses normalized.png.
        raise AppError("not_found", f"avatar {avatar_id} has no idle video", 404)
    root = store.avatar_dir(avatar_id)
    path = root / "idle.mp4"
    if not path.exists():
        # Built lazily on this threadpool worker: pure CPU, one-time per avatar.
        try:
            from server.avatar.tunables import effective

            # Pull crf/preset at build time, not at module import time, so a
            # operator PATCH from another console wins the next idle rebuild.
            # The per-avatar overrides (set by the create form) beat the
            # operator-pushed Setting rows so this avatar's quality knob
            # survives the build.
            settings = effective(db, overrides=avatar.tunable_overrides or None)
            path = video.build_idle_video(
                root,
                root / pack.prep.get("source", "source.mp4"),
                get_settings(),
                crf=int(settings["idle_crf"]),
                preset=str(settings["idle_preset"]),
            )
        except video.VideoError as exc:
            raise AppError("idle_video_unavailable", str(exc), 503) from exc
    return FileResponse(path, media_type="video/mp4")


@router.get("/avatars/{avatar_id}/normalized.png")
def avatar_normalized_png(avatar_id: str, db=Depends(get_db)):
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    path = Path(pack.root) / pack.prep.get("base_image", "normalized.png")
    if not path.exists():
        raise AppError("not_found", "avatar image not found", 404)
    return FileResponse(path, media_type="image/png")


@router.get("/avatars/{avatar_id}/artifacts")
def avatar_artifacts(avatar_id: str, db=Depends(get_db)) -> list[dict]:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return _list_avatar_artifacts(avatar)


@router.get("/avatars/{avatar_id}/timeline")
def avatar_timeline(avatar_id: str, db=Depends(get_db)) -> dict:
    """Surface the pipeline's per-stage meta so the UI can show how this
    avatar was built (which face crop, which detector, frame counts, quality
    warnings) without parsing ``pack.json`` itself.
    """
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    raw_meta: dict | None = None
    meta_path = AvatarStore(get_settings().data_dir).avatar_dir(avatar_id) / "musetalk" / "meta.json"
    if meta_path.exists():
        try:
            raw_meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            raw_meta = None
    return _pipeline_timeline(avatar, raw_meta)


@router.get("/avatars/{avatar_id}/artifact/{path:path}")
def avatar_artifact_file(avatar_id: str, path: str, db=Depends(get_db)):
    """Stream a single artifact's bytes (the listing endpoint returns metadata
    only; this endpoint serves the contents).

    Path-traversal guard: the avatar id is uuid-hex (already validated by
    ``_require_uuid_hex``); the artifact path is checked against the avatar's
    on-disk root before any read.
    """
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    root = AvatarStore(get_settings().data_dir).avatar_dir(avatar_id).resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AppError("invalid_path", "artifact path escapes avatar root", 400) from exc
    if not candidate.exists() or not candidate.is_file():
        raise AppError("not_found", f"artifact {path} not found", 404)
    media = "application/octet-stream"
    name = candidate.name.lower()
    if name.endswith(".png") or name.endswith(".jpg"):
        media = f"image/{name.rsplit('.', 1)[-1]}"
    elif name.endswith(".mp4"):
        media = "video/mp4"
    elif name.endswith(".json"):
        media = "application/json"
    return FileResponse(candidate, media_type=media)


@router.post("/tts")
async def synthesize_speech(body: TtsIn) -> Response:
    """Turn typed text into WAV audio for the console's debug page.

    Session-agnostic on purpose: the console pushes the bytes it gets back down
    the same audio WebSocket the file-upload path uses, so this endpoint never
    needs to know which session is being driven.
    """
    text = body.text.strip()
    if not text:
        raise AppError("empty_text", "text must not be empty", 400)
    from server.tts.minimax import synthesize

    audio = await synthesize(text, body.voice_id)
    return Response(audio, media_type="audio/wav")


@router.get("/jobs")
def list_jobs(db=Depends(get_db)) -> list[dict]:
    rows = (
        db.execute(select(AvatarJob).order_by(AvatarJob.created_at.desc()).limit(200)).scalars().all()
    )
    return [
        {
            "job_id": j.id,
            "avatar_id": j.avatar_id,
            "stage": j.stage,
            "progress": j.progress,
            "error": j.error,
            "created_at": j.created_at.isoformat(),
        }
        for j in rows
    ]


@router.get("/sessions")
def admin_list_sessions(db=Depends(get_db)) -> list[dict]:
    from server.main import get_session_manager

    manager = get_session_manager()
    rows = (
        db.execute(select(StreamSession).order_by(StreamSession.started_at.desc()).limit(100))
        .scalars()
        .all()
    )
    out = []
    for row in rows:
        render = manager._renders.get(row.id)
        metrics = render.metrics if render else (row.metrics_json or {})
        out.append(
            {
                "session_id": row.id,
                "avatar_id": row.avatar_id,
                "status": render.state.value if render else row.status,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "closed_at": row.closed_at.isoformat() if row.closed_at else None,
                "metrics": metrics,
            }
        )
    return out


@router.post("/sessions/sweep")
def admin_sweep(db=Depends(get_db)) -> dict:
    """Purge every StreamSession row whose status is 'closed'.

    The lifespan sweeper force-closes idle sessions by flipping the row's
    status without deleting it; over time that leaves a long tail in the
    admin list. This endpoint drops those rows so the System page list
    shrinks after a sweep. Returns the count for the UI toast.
    """
    closed = db.execute(
        sa_delete(StreamSession).where(StreamSession.status == "closed")
    ).rowcount
    db.commit()
    return {"closed": closed}


@router.delete("/sessions/{session_id}", status_code=204)
def admin_close_session(session_id: str, db=Depends(get_db)) -> Response:
    """Drop a session row from the admin list.

    Closes any live render first (so an active conversation is also
    unwound) and then deletes the StreamSession row. Active sessions
    disappear from the list immediately; the prior "force-close but
    keep the row" behaviour left a long tail of closed rows the
    operator had to live with.
    """
    import asyncio

    from server.main import get_session_manager

    row = db.get(StreamSession, session_id)
    if row is None:
        raise AppError("not_found", f"session {session_id} not found", 404)
    manager = get_session_manager()
    # Awaiting close() inside asyncio.run is safe: the per-session
    # RenderSession holds no reference back to the request loop.
    asyncio.run(manager.close(db, session_id))
    # close() may have transitioned status to "closed" without removing
    # the row (it never deletes), so delete here unconditionally.
    db.delete(row)
    db.commit()
    return Response(status_code=204)


# ----- Models: load / unload -----------------------------------------------


def _active_session_count(db) -> int:
    """How many streaming sessions are currently *running*.

    The Models tab uses this to refuse ``unload`` under live traffic --
    tearing the engine runtime out from under a session makes its next
    ``drive()`` pay the ~30 s cold load. Same definition the lifespan
    sweeper uses so the unload gate and the sweep tick never disagree.
    """
    from sqlalchemy import func

    return int(
        db.execute(
            select(func.count())
            .select_from(StreamSession)
            .where(StreamSession.status == "active")
        ).scalar_one()
    )


def _resolve_engine_slot(engine, model_id: str) -> dict | None:
    """Look up the engine model slot by id. Returns ``None`` if missing.

    Used by the load/unload routes so a typo'd id is a clean 404, not a
    500. The lookup is read-only; the response shape matches the row in
    ``describe()["models"]`` so the caller can use the slot's ``loaded``
    + ``can_unload`` flags without re-reading describe().
    """
    described = engine.describe()
    for row in described.get("models") or []:
        if row.get("id") == model_id:
            return row
    return None


# ----- service surface (animators + TTS) -----------------------------------

# Animators register into ``ANIMATORS`` via side-effect imports at module
# load time (the ``# noqa: F401`` lines at the top of this file). Each
# entry below reads its meta() + is_ready + on-disk weights path so the
# Models tab can render every backend this server ships with in one row
# per entry, even though the backends themselves are loaded on demand
# during a single ``animate()`` call rather than held long-term.

#: The display key for a TTS row in the services list.
_TTS_ID = "tts:minimax"


def _animator_service_row(animator) -> dict | None:
    """Render one portrait animator as a Models tab row.

    Each backend (LivePortrait / Wan 2.2 / MuseTalk silent) has its own
    weights directory and (for Wan 2.2) multiple variants. The Models
    tab aggregates them into a single list so the operator can confirm
    "which backends are wired in right now" at a glance -- without
    flipping to the 形象活化 tab.

    Returns ``None`` only when the backend is unregistered (rare); a
    broken / unloaded backend still appears, with ``configured=False``
    + a ``missing_script`` hint at the fetch script that would fix it.

    Each row also carries ``loaded`` / ``can_load`` / ``can_unload``
    so the operator can pre-allocate the backend's pipeline from the
    Models tab. ``can_load`` is True when the backend has its
    pre-allocation method (liveportrait + wan22 + musetalk-silent --
    all three gained a ``load()`` for the operator-driven lifecycle);
    ``can_unload`` mirrors ``loaded``: a row that's already in VRAM
    can release it.
    """
    from server.avatar.vram import get_vram_gb

    try:
        meta = animator.meta()
    except Exception:  # noqa: BLE001 -- meta() can probe torch; failure == unregistered
        return None

    try:
        ready = bool(animator.is_ready)
    except Exception:  # noqa: BLE001 -- readiness probe must never raise
        ready = False

    weights_dir = None
    for attr in ("weights_dir", "WEIGHTS_ROOT", "_weights_root"):
        candidate = getattr(animator, attr, None)
        if candidate is not None:
            try:
                weights_dir = str(Path(str(candidate)))
            except Exception:  # noqa: BLE001
                weights_dir = None
            break

    weights_present = False
    if weights_dir:
        try:
            wd = Path(weights_dir)
            weights_present = wd.is_dir() and any(wd.iterdir())
        except OSError:
            weights_present = False

    # The lifecycle flags surface the operator's two questions: "is
    # this on the GPU right now?" (``loaded``) and "can the Models tab
    # release it?" (``can_unload``).
    #
    # ``can_load`` keys on "the backend exposes a ``load()`` method"
    # -- NOT on ``weights_present``. The reason: animators like
    # MuseTalk silent piggyback on the main drive engine, so they
    # have no on-disk weights of their own; gating on weights_present
    # would render their 加载 button as "无可用操作" forever, with no
    # way for the operator to even trigger the 503 that would tell
    # them where the real problem lies. ``load()`` itself surfaces the
    # right error (503 with the fetch-script hint) when it fails.
    try:
        loaded = bool(getattr(animator, "is_loaded", False))
    except Exception:  # noqa: BLE001
        loaded = False
    can_load = bool(getattr(animator, "load", None))
    can_unload = loaded

    script = None
    cid = getattr(animator, "name", "")
    if not ready:
        if cid == "liveportrait":
            script = "scripts/fetch_liveportrait_weights.sh"
        elif cid.startswith("wan22-"):
            script = "scripts/fetch_wan22_weights.sh"
        elif cid == "musetalk-silent":
            script = None

    # Status string covers four states the operator cares about:
    # weights missing (fetch script), dependencies missing
    # (no script, run install), ready (unloaded), and live (loaded).
    if not weights_present and script:
        status = "无权重"
    elif not ready:
        status = "环境异常"
    elif loaded:
        status = "已加载"
    else:
        status = "可加载"

    return {
        "id": f"animator:{cid}",
        "kind": "portrait_animator",
        "label_zh": meta.get("label_zh", cid),
        "group_zh": meta.get("group_zh", ""),
        "description": meta.get("description", ""),
        "vram_required_gb": meta.get("vram_required_gb"),
        "vram_hint": meta.get("vram_hint", ""),
        "fits_current_gpu": meta.get("fits_current_gpu"),
        "configured": ready,
        "weights_present": weights_present,
        "weights_dir": weights_dir,
        "missing_script": script,
        "loaded": loaded,
        "can_load": can_load,
        "can_unload": can_unload,
        # Status covers four operator-visible states (see above).
        "status": status,
    }


def _animator_service_rows() -> list[dict]:
    """Walk ANIMATORS + Wan 2.2 variants and render each as a row.

    ``ANIMATORS`` is the canonical map populated by side-effect imports
    at module load. Wan 2.2 isn't in there -- each variant is a fresh
    instance built by ``all_animators()`` -- so we walk both for
    exhaustive coverage.
    """
    out: list[dict] = []
    try:
        from server.avatar.animators import ANIMATORS
    except Exception:  # noqa: BLE001 -- registry import may fail in stub venvs
        ANIMATORS = {}

    seen_ids: set[str] = set()
    for name, animator in ANIMATORS.items():
        row = _animator_service_row(animator)
        if row is not None:
            out.append(row)
            seen_ids.add(row["id"])

    # Wan 2.2 variants are not in ANIMATORS -- they're constructed per
    # request by ``all_animators()``. Walk the variants directly so the
    # Models tab sees each one as its own row.
    try:
        from server.avatar import wan22

        for animator in wan22.all_animators():
            row = _animator_service_row(animator)
            if row is not None and row["id"] not in seen_ids:
                out.append(row)
                seen_ids.add(row["id"])
    except Exception:  # noqa: BLE001 -- wan22 may not be importable in stub venvs
        pass

    return out


def _tts_service_row() -> dict:
    """MiniMax TTS as a Models tab row.

    TTS isn't a local model -- the audio comes from api.minimaxi.com --
    but the operator still wants to see "is it configured, what's the
    voice, can I synthesise at all" on the Models tab. Local weight
    presence is replaced with ``configured`` + voice id + sample rate cap,
    all read from settings.
    """
    from server.config import get_settings

    settings = get_settings()
    configured = bool(getattr(settings, "minimax_api_key", ""))
    return {
        "id": _TTS_ID,
        "kind": "tts_service",
        "label_zh": "MiniMax TTS",
        "group_zh": "云端服务",
        "description": (
            "数字人对话使用 MiniMax 的语音合成接口；"
            "需要配置 DH_MINIMAX_API_KEY；采样率上限 44100 Hz"
        ),
        "configured": configured,
        "weights_present": False,  # remote service, no on-disk weights
        "weights_dir": None,
        "missing_script": None if configured else "set DH_MINIMAX_API_KEY in .env",
        "vram_required_gb": 0,
        "vram_hint": "云端",
        "fits_current_gpu": None,
        "voice_id": getattr(settings, "minimax_voice_id", ""),
        "model": getattr(settings, "minimax_model", "speech-02-hd"),
        "sample_rate_cap_hz": 44100,
        "sample_rate": getattr(settings, "minimax_sample_rate", 32000),
        "status": "已配置" if configured else "未配置",
    }


@router.get("/models")
def admin_list_models(db=Depends(get_db)) -> dict:
    """Surface the live drive-engine runtime state for the Models tab.

    Single payload the page polls every 2 s. Returns:

    * ``models`` -- one entry per engine slot (today: a single
      ``drive_runtime``). Each entry carries its ``loaded`` flag plus a
      per-component breakdown so the operator can see "what is on the
      GPU right now".
    * ``services`` -- one entry per every other model this server uses:
      portrait animators (LivePortrait / Wan 2.2 variants / MuseTalk
      silent) and the MiniMax TTS endpoint. These aren't held long-term
      so they have no ``loaded`` / ``can_unload`` flags; instead they
      expose a ``configured`` flag + on-disk weights directory so the
      operator can spot a missing-weight backend without flipping tabs.
    * ``gpu`` -- a :func:`get_vram_snapshot` dict or ``None`` when CUDA
      isn't available (the tab renders a CPU placeholder instead).
    * ``active_sessions`` -- count of currently-running StreamSession
      rows; the unload button refuses when this is non-zero.
    """
    from server.avatar.vram import get_vram_snapshot
    from server.main import get_session_manager

    engine = get_session_manager().engine
    described = engine.describe()
    return {
        "models": described.get("models") or [],
        "services": _animator_service_rows() + [_tts_service_row()],
        "gpu": get_vram_snapshot(),
        "active_sessions": _active_session_count(db),
    }


@router.post("/models/{model_id}/load")
async def admin_load_model(model_id: str) -> dict:
    """Trigger the engine's ``load`` for the named slot.

    Routes to ``engine.load()`` if exposed; falls back to a 400 for
    engines with nothing material to load (mock). 409 if the slot is
    already loaded, 503 if loading raises an ``EngineUnavailable`` (e.g.
    weights missing on disk).
    """
    from server.main import get_session_manager

    engine = get_session_manager().engine
    slot = _resolve_engine_slot(engine, model_id)
    if slot is None:
        raise AppError("not_found", f"model {model_id!r} not found", 404)
    if slot.get("loaded"):
        raise AppError(
            "already_loaded",
            f"model {model_id!r} is already loaded",
            409,
            detail={"id": model_id},
        )
    loader = getattr(engine, "load", None)
    if loader is None:
        raise AppError(
            "no_weights_to_load",
            f"engine {engine.name!r} has no weights to load",
            400,
            detail={"engine": engine.name},
        )
    try:
        # Real engine.load() may touch torch; run on the engine's own
        # worker so the event loop isn't blocked for ~30 s on a cold load.
        maybe = loader()
        if hasattr(maybe, "__await__"):
            await maybe
    except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
        # EngineUnavailable from warmup() means weights missing; mirror its
        # production log surface so the operator sees the same hint.
        raise AppError(
            "load_failed",
            f"{exc}",
            503,
            detail={"engine": engine.name, "id": model_id},
        )
    return {"id": model_id, "loaded": True}


@router.post("/models/{model_id}/unload")
def admin_unload_model(model_id: str, db=Depends(get_db)) -> dict:
    """Free the engine runtime for the named slot.

    Refuses with 409 while any session is active so tearing the runtime
    out doesn't silently wedge live conversations; refuses with 400 when
    the engine reports ``can_unload=False`` (mock); 404 for an unknown id.
    """
    from server.main import get_session_manager

    engine = get_session_manager().engine
    slot = _resolve_engine_slot(engine, model_id)
    if slot is None:
        raise AppError("not_found", f"model {model_id!r} not found", 404)
    if not slot.get("can_unload"):
        raise AppError(
            "not_unloadable",
            f"model {model_id!r} has no weights to unload",
            400,
            detail={"engine": engine.name, "id": model_id},
        )
    if not slot.get("loaded"):
        # Idempotent: already unloaded. Surface the same way as load's
        # "already_loaded" 409 so the UI can show a clear message instead
        # of a confusing 200.
        raise AppError(
            "not_loaded",
            f"model {model_id!r} is not loaded",
            409,
            detail={"id": model_id},
        )
    busy = _active_session_count(db)
    if busy > 0:
        # Operator is told how many active sessions block the unload -- they
        # go to the 会话 tab and clean up, then come back.
        raise AppError(
            "engine_busy",
            f"engine has {busy} active session(s); close them before unloading",
            409,
            detail={"id": model_id, "active_sessions": busy},
        )
    engine.unload()
    return {"id": model_id, "loaded": False}


# ----- service load / unload (animators) -----------------------------------


def _resolve_animator_by_id(service_id: str):
    """Find an ANIMATORS entry by the Models-tab id ``animator:<name>``.

    Returns ``None`` when the prefix doesn't match or the backend is
    unregistered (the typo case). TTS services have their own id
    pattern (``tts:minimax``) and don't go through here -- they have
    no operator-loadable lifecycle.
    """
    prefix = "animator:"
    if not service_id.startswith(prefix):
        return None
    name = service_id[len(prefix):]
    try:
        from server.avatar.animators import ANIMATORS
    except Exception:  # noqa: BLE001 -- registry import may fail in stub venvs
        return None
    return ANIMATORS.get(name)


@router.post("/services/{service_id}/load")
async def admin_load_service(service_id: str) -> dict:
    """Trigger ``load()`` on the named service.

    Dispatches to the appropriate backend's pre-allocation method.
    Returns the same 200/409/503 semantics as the drive-engine
    ``models/.../load`` route so the operator gets a uniform error
    surface across the Models tab.

    * 200 -- service moved from ``loaded=False`` to ``loaded=True``.
    * 409 -- already loaded (idempotency rejected so the operator can
      tell the click was processed).
    * 400 -- service has no weights to load (mock or remote TTS).
    * 503 -- load raised an ``AnimatorError`` (missing runtime /
      weights -- the operator must run the fetch script).
    """
    animator = _resolve_animator_by_id(service_id)
    if animator is None:
        raise AppError("not_found", f"service {service_id!r} not found", 404)

    # ``is_loaded`` is the source of truth across all three backends:
    # LivePortrait / Wan read it from their own flag; MuseTalk silent
    # proxies to the main drive engine's runtime.
    try:
        loaded = bool(getattr(animator, "is_loaded", False))
    except Exception:  # noqa: BLE001
        loaded = False

    if loaded:
        raise AppError(
            "already_loaded",
            f"service {service_id!r} is already loaded",
            409,
            detail={"id": service_id},
        )

    loader = getattr(animator, "load", None)
    if loader is None:
        raise AppError(
            "no_weights_to_load",
            f"service {service_id!r} has no weights to load",
            400,
            detail={"id": service_id},
        )

    try:
        result = loader()
        if hasattr(result, "__await__"):
            await result
    except Exception as exc:  # noqa: BLE001 -- surface verbatim to the operator
        raise AppError(
            "load_failed",
            f"{exc}",
            503,
            detail={"id": service_id, "engine": getattr(animator, "name", service_id)},
        )
    return {"id": service_id, "loaded": True}


@router.post("/services/{service_id}/unload")
def admin_unload_service(service_id: str) -> dict:
    """Release any VRAM the named service is holding.

    Mirror image of the load route. ``is_loaded`` is the gate: a
    not-loaded service returns 409 ``not_loaded`` so the operator can
    tell the click was processed without a confusing 200-no-op.

    Services without an ``unload`` method (the future TTS-shaped
    entrypoints) return 400 ``not_unloadable`` instead of 200, so the
    operator's button is correctly disabled on the client side and
    any race gives a clear error.
    """
    animator = _resolve_animator_by_id(service_id)
    if animator is None:
        raise AppError("not_found", f"service {service_id!r} not found", 404)

    try:
        loaded = bool(getattr(animator, "is_loaded", False))
    except Exception:  # noqa: BLE001
        loaded = False

    if not loaded:
        raise AppError(
            "not_loaded",
            f"service {service_id!r} is not loaded",
            409,
            detail={"id": service_id},
        )

    unloader = getattr(animator, "unload", None)
    if unloader is None:
        raise AppError(
            "not_unloadable",
            f"service {service_id!r} has no unload method",
            400,
            detail={"id": service_id},
        )
    unloader()
    return {"id": service_id, "loaded": False}
