import re
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from starlette.responses import Response

from server.avatar.pipeline import run_job_in_new_session
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob
from server.db.session import get_db

router = APIRouter(prefix="/api/v1/avatars", tags=["avatars"])

_UUID_HEX = re.compile(r"\A[0-9a-f]{32}\Z")


def _store() -> AvatarStore:
    return AvatarStore(get_settings().data_dir)


def _require_uuid_hex(avatar_id: str) -> str:
    # Path-traversal guard: ids that reach the filesystem must be our uuid4().hex.
    if not _UUID_HEX.match(avatar_id):
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return avatar_id


def _validate_suffix(suffix: str) -> str:
    # Path-traversal guard: the suffix becomes part of a filename under the avatar dir.
    if any(sep in suffix for sep in ("/", "\\")) or ".." in suffix:
        raise AppError("invalid_filename", "filename must not contain path separators or '..'", 400)
    return suffix


def _derive_suffix(filename: str | None) -> str:
    ext = filename.rsplit(".", 1)[-1] if filename and "." in filename else "png"
    return _validate_suffix("." + ext)


def _serialize(a: Avatar, with_detail: bool = False) -> dict:
    data = {
        "avatar_id": a.id,
        "name": a.name,
        "status": a.status,
        "created_at": a.created_at.isoformat(),
    }
    if with_detail:
        data["config"] = a.config_json
        data["error"] = None
    return data


def _create_avatar_rows(
    db,
    name: str,
    stream,
    suffix: str,
    tunable_overrides: dict | None = None,
) -> tuple[str, str]:
    """Create the Avatar + AvatarJob rows and return ``(avatar_id, job_id)``.

    Shared by the upload endpoint here and the admin upload endpoint in
    ``server.api.admin`` so neither duplicates the store/row wiring. Scheduling
    the background job stays with each caller (each owns its BackgroundTasks).

    ``tunable_overrides`` is the per-avatar form of the tunables that the
    admin create form captures (idle_crf / idle_preset). ``None`` (or ``{}``)
    means "use the global defaults"; the column is NOT NULL with a JSON
    default so a non-admin upload does not need to set anything.
    """
    avatar_id = uuid4().hex
    store = _store()
    source_path = store.save_source(avatar_id, stream, suffix)

    avatar = Avatar(
        id=avatar_id,
        name=name,
        status="pending",
        source_image_path=str(source_path),
        config_json={},
        tunable_overrides=tunable_overrides or {},
    )
    job = AvatarJob(id=uuid4().hex, avatar_id=avatar_id, stage="queued")
    db.add(avatar)
    db.add(job)
    db.commit()
    return avatar_id, job.id


@router.post("", status_code=201)
def create_avatar(
    background: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = ...,
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
    avatar_id, job_id = _create_avatar_rows(db, name, file.file, suffix)
    background.add_task(run_job_in_new_session, job_id, _store())
    return {"avatar_id": avatar_id, "status": "pending"}


@router.get("")
def list_avatars(db=Depends(get_db)) -> list[dict]:
    rows = (
        db.execute(select(Avatar).order_by(Avatar.created_at.desc()))
        .scalars()
        .all()
    )
    return [_serialize(a) for a in rows]


@router.get("/{avatar_id}")
def get_avatar(avatar_id: str, db=Depends(get_db)) -> dict:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    data = _serialize(avatar, with_detail=True)
    if avatar.status == "failed":
        job = (
            db.execute(
                select(AvatarJob)
                .where(AvatarJob.avatar_id == avatar_id)
                .order_by(AvatarJob.created_at.desc())
            )
            .scalars()
            .first()
        )
        data["error"] = job.error if job else "unknown error"
    return data


@router.delete("/{avatar_id}", status_code=204)
def delete_avatar(avatar_id: str, db=Depends(get_db)) -> Response:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    db.execute(sa_delete(AvatarJob).where(AvatarJob.avatar_id == avatar_id))
    db.delete(avatar)
    db.commit()
    _store().delete_assets(avatar_id)
    return Response(status_code=204)