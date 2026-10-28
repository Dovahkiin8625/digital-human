import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from dh_server.avatar.store import AvatarStore


class PipelineError(Exception):
    pass


class AvatarPipeline(ABC):
    """Turns an uploaded source into the pack the model drives.

    ``inline_build`` says whether the build is cheap enough to finish
    inside the upload request instead of a background job: the Live2D
    pipeline extracts and validates a zip, while MuseTalk's runs for
    minutes on the GPU.
    """

    inline_build: bool = False

    @abstractmethod
    def process(
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress: Callable[[str, int], None],
        *,
        options: dict | None = None,
    ) -> dict:
        """Build the avatar's pack in ``out_dir`` and return its config.

        ``options`` carries the per-model extras the operator filled in
        on the create form (Live2D's entry point and emotion map). A
        model that needs none ignores it.
        """


def create_pipeline(name: str | None = None) -> AvatarPipeline:
    """Build the avatar pipeline of a registered digital-human model.

    ``name`` defaults to the fallback model (see
    ``dh_server.model_registry``). The pipeline always comes from the model
    that owns the avatar, so a pack can never be written by one model and
    driven by another.

    Raises :class:`PipelineError` for a name that is not registered.
    """
    from dh_server.model_registry import ModelUnavailable, get_model

    try:
        model = get_model(name)
    except ModelUnavailable as exc:
        raise PipelineError(str(exc)) from exc
    return model.pipeline_factory()


def run_avatar_job(db, store, pipeline: AvatarPipeline, job_id: str) -> None:
    from dh_server.db.models import Avatar, AvatarJob

    job = db.get(AvatarJob, job_id)
    if job is None:
        return
    avatar = db.get(Avatar, job.avatar_id)
    if avatar is None:
        return

    def progress(stage: str, pct: int) -> None:
        job.stage = stage
        job.progress = pct
        db.commit()

    try:
        avatar.status = "processing"
        db.commit()
        from dh_server.avatar.tunables import effective

        # The per-avatar overrides the create form captured beat the module
        # defaults. The audit row reflects exactly what this build saw, so
        # a later idle-video re-encode (which reads from the same column)
        # will produce the same result.
        effective_now = effective(overrides=avatar.tunable_overrides or None)
        config = pipeline.process(
            avatar.id,
            Path(avatar.source_image_path),
            store.avatar_dir(avatar.id),
            progress,
            # The per-avatar build options the create form captured. The
            # models that need them read their own keys; the others ignore
            # them.
            options=avatar.build_options or None,
        )
        pack_json = store.avatar_dir(avatar.id) / "pack.json"
        pack_json.write_text(json.dumps(config))
        # Audit: write the tunables that applied to this build, so the
        # avatar's detail page can show what it was built with.
        config["_effective_params"] = effective_now
        avatar.config_json = config
        avatar.status = "ready"
        db.commit()
    except Exception as exc:  # noqa: BLE001 — 失败必须落库可见
        db.rollback()
        avatar.status = "failed"
        job.error = str(exc)[:1024]
        db.commit()


def create_avatar_rows(
    db,
    name: str,
    stream,
    suffix: str,
    tunable_overrides: dict | None = None,
    drive_model: str | None = None,
    *,
    build_options: dict | None = None,
    max_video_seconds: float | None = None,
    max_video_frames: int | None = None,
) -> tuple[str, str]:
    """Create the Avatar and AvatarJob rows, and return their ids.

    The source file is written to the avatar's directory first, so a row
    never points at a file that is not there. Scheduling the build stays
    with the caller: this function only records what has to be built.

    ``tunable_overrides`` is the per-avatar form of the tunables that the
    create form captures (idle_crf / idle_preset). ``None`` (or ``{}``)
    means "use the global defaults"; the column is NOT NULL with a JSON
    default so nothing has to set it.

    ``drive_model`` names the digital-human model that will drive this
    avatar. It is recorded on the row, so the pack this job builds and the
    engine that later drives it always belong to the same model.

    ``build_options`` is handed to the model's pipeline when the job runs.
    It lives on the avatar row rather than on the job so a retry rebuilds
    the avatar the same way.

    ``max_video_seconds`` / ``max_video_frames`` override the global upload
    limits for models whose source is longer than the default.
    """
    from dh_server.avatar.store import AvatarStore
    from dh_server.config import get_settings
    from dh_server.db.models import Avatar, AvatarJob
    from dh_server.model_registry import DEFAULT_MODEL

    avatar_id = uuid4().hex
    store = AvatarStore(get_settings().data_dir)
    source_path = store.save_source(
        avatar_id,
        stream,
        suffix,
        max_seconds=max_video_seconds,
        max_frames=max_video_frames,
    )

    avatar = Avatar(
        id=avatar_id,
        name=name,
        status="pending",
        source_image_path=str(source_path),
        config_json={},
        tunable_overrides=tunable_overrides or {},
        build_options=build_options or {},
        drive_model=drive_model or DEFAULT_MODEL,
    )
    job = AvatarJob(id=uuid4().hex, avatar_id=avatar_id, stage="queued")
    db.add(avatar)
    db.add(job)
    db.commit()
    return avatar_id, job.id


def run_job_in_new_session(job_id: str, store: "AvatarStore") -> None:
    """Run an avatar job to completion in its own DB session.

    Shared by the FastAPI BackgroundTasks call sites in api/admin.py (the
    upload and the retry), so both run the same job runner.

    The pipeline comes from the model that owns the avatar, so the pack is
    always written by the model that will drive it.
    """
    from sqlalchemy.orm import sessionmaker

    from dh_server.db.models import Avatar, AvatarJob
    from dh_server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        job = db.get(AvatarJob, job_id)
        avatar = db.get(Avatar, job.avatar_id) if job is not None else None
        if avatar is None:
            return
        run_avatar_job(db, store, create_pipeline(avatar.drive_model), job_id)
    finally:
        db.close()
