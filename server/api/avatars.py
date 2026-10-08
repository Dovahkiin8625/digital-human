import re
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from starlette.responses import Response

from server.api.deps import get_tenant
from server.avatar.pipeline import create_pipeline, run_avatar_job
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob, Tenant
from server.db.session import get_db, get_engine

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


def _get_owned(db, tenant: Tenant, avatar_id: str) -> Avatar:
    avatar = db.execute(
        select(Avatar).where(Avatar.id == avatar_id, Avatar.tenant_id == tenant.id)
    ).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return avatar


@router.post("", status_code=201)
def create_avatar(
    background: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = ...,
    tenant: Tenant = Depends(get_tenant),
    db=Depends(get_db),
) -> dict:
    avatar_id = uuid4().hex
    store = _store()
    data = file.file.read()
    suffix = _derive_suffix(file.filename)
    try:
        source_path = store.save_source_image(avatar_id, data, suffix)
    except Exception as exc:
        raise AppError("invalid_image", str(exc), 422) from exc

    avatar = Avatar(
        id=avatar_id,
        tenant_id=tenant.id,
        name=name,
        status="pending",
        source_image_path=str(source_path),
    )
    job = AvatarJob(avatar_id=avatar_id, stage="queued")
    db.add(avatar)
    db.add(job)
    db.commit()
    background.add_task(_run_job, job.id)
    return {"avatar_id": avatar_id, "status": "pending"}


def _run_job(job_id: str) -> None:
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    try:
        run_avatar_job(db, _store(), create_pipeline("mock"), job_id)
    finally:
        db.close()


@router.get("")
def list_avatars(tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> list[dict]:
    rows = (
        db.execute(
            select(Avatar).where(Avatar.tenant_id == tenant.id).order_by(Avatar.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_serialize(a) for a in rows]


@router.get("/{avatar_id}")
def get_avatar(avatar_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    _require_uuid_hex(avatar_id)
    avatar = _get_owned(db, tenant, avatar_id)
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
def delete_avatar(avatar_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> Response:
    _require_uuid_hex(avatar_id)
    avatar = _get_owned(db, tenant, avatar_id)
    db.execute(sa_delete(AvatarJob).where(AvatarJob.avatar_id == avatar_id))
    db.delete(avatar)
    db.commit()
    _store().delete_assets(avatar_id)
    return Response(status_code=204)
