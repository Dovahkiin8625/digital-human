import hmac
import json
from pathlib import Path
from secrets import token_urlsafe
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from starlette.responses import FileResponse, Response

from server.api.avatars import _create_avatar_rows, _derive_suffix, _require_uuid_hex
from server.api.deps import get_admin, hash_api_key
from server.avatar.pipeline import run_job_in_new_session
from server.avatar import video
from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar, AvatarJob, StreamSession, Tenant
from server.db.session import get_db

# Split routers: login is the auth entry point and must NOT sit behind get_admin;
# every other admin endpoint does (controller ruling #2).
router = APIRouter(prefix="/api/admin", tags=["admin"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(get_admin)])


class LoginIn(BaseModel):
    password: str


class TenantIn(BaseModel):
    name: str


class AvatarConfigIn(BaseModel):
    mouth_scale: float | None = None
    expression_intensity: float | None = None


class TtsIn(BaseModel):
    text: str
    voice_id: str | None = None


@router.post("/login")
def login(body: LoginIn) -> dict:
    expected = get_settings().admin_password
    if not hmac.compare_digest(body.password.encode("utf-8"), expected.encode("utf-8")):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
    return {"token": get_settings().admin_token}


@admin_router.get("/system")
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
    }


@admin_router.get("/tenants")
def list_tenants(db=Depends(get_db)) -> list[dict]:
    rows = db.execute(select(Tenant).order_by(Tenant.created_at.desc())).scalars().all()
    return [
        {"tenant_id": t.id, "name": t.name, "created_at": t.created_at.isoformat()} for t in rows
    ]


@admin_router.post("/tenants", status_code=201)
def create_tenant(body: TenantIn, db=Depends(get_db)) -> dict:
    api_key = "dh_" + token_urlsafe(32)
    tenant = Tenant(name=body.name, api_key_hash=hash_api_key(api_key))
    db.add(tenant)
    db.commit()
    return {"tenant_id": tenant.id, "name": tenant.name, "api_key": api_key}


@admin_router.post("/tenants/{tenant_id}/api-key")
def issue_tenant_api_key(tenant_id: str, db=Depends(get_db)) -> dict:
    """Issue (rotate) the tenant's API key and return the new plaintext once.

    Keys are stored only as a hash, so an existing key cannot be re-read; issuing
    is the only way to hand a key to the console's demo-key bootstrap. Rotation
    replaces the hash, invalidating the previous key.
    """
    tenant = db.execute(select(Tenant).where(Tenant.id == tenant_id)).scalar_one_or_none()
    if tenant is None:
        raise AppError("not_found", f"tenant {tenant_id} not found", 404)
    api_key = "dh_" + token_urlsafe(32)
    tenant.api_key_hash = hash_api_key(api_key)
    db.commit()
    return {"tenant_id": tenant.id, "api_key": api_key}


@admin_router.get("/avatars")
def admin_list_avatars(tenant_id: str = "", db=Depends(get_db)) -> list[dict]:
    stmt = select(Avatar)
    if tenant_id:
        stmt = stmt.where(Avatar.tenant_id == tenant_id)
    rows = db.execute(stmt.order_by(Avatar.created_at.desc())).scalars().all()
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
        "tenant_id": a.tenant_id,
        "name": a.name,
        "status": a.status,
        "config": a.config_json,
        "error": job.error if a.status == "failed" and job else None,
        "created_at": a.created_at.isoformat(),
    }


def _get_or_create_default_tenant(db) -> Tenant:
    """The designated tenant for admin-created avatars when no tenant_id is given.

    Deliberately the tenant named "default" (not merely the oldest): the session
    debug page issues an API key for this same tenant, and create_session requires
    the avatar to belong to that key's tenant, so both surfaces must agree.
    """
    tenant = db.execute(select(Tenant).where(Tenant.name == "default")).scalars().first()
    if tenant is None:
        tenant = Tenant(name="default", api_key_hash=hash_api_key("dh_" + token_urlsafe(32)))
        db.add(tenant)
        db.commit()
    return tenant


@admin_router.post("/avatars", status_code=201)
def admin_create_avatar(
    background: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = ...,
    tenant_id: str = Form(""),
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
    if not tenant_id:
        tenant_id = _get_or_create_default_tenant(db).id
    avatar_id, job_id = _create_avatar_rows(db, tenant_id, name, file.file, suffix)
    background.add_task(run_job_in_new_session, job_id, AvatarStore(get_settings().data_dir))
    return {"avatar_id": avatar_id, "status": "pending"}


@admin_router.get("/avatars/{avatar_id}")
def admin_get_avatar(avatar_id: str, db=Depends(get_db)) -> dict:
    _require_uuid_hex(avatar_id)
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    return _admin_avatar_row(db, avatar)


@admin_router.delete("/avatars/{avatar_id}", status_code=204)
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


@admin_router.post("/avatars/{avatar_id}/retry", status_code=202)
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


@admin_router.patch("/avatars/{avatar_id}/config")
def patch_avatar_config(avatar_id: str, body: AvatarConfigIn, db=Depends(get_db)) -> dict:
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    cfg = dict(avatar.config_json or {})
    if body.mouth_scale is not None:
        cfg["mouth_scale"] = body.mouth_scale
    if body.expression_intensity is not None:
        cfg["expression_intensity"] = body.expression_intensity
    avatar.config_json = cfg
    if avatar.pack_path:
        Path(avatar.pack_path).write_text(json.dumps(cfg))
    db.commit()
    return {"avatar_id": avatar_id, "config": cfg}


@admin_router.get("/avatars/{avatar_id}/idle.mp4")
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
            path = video.build_idle_video(
                root, root / pack.prep.get("source", "source.mp4"), get_settings()
            )
        except video.VideoError as exc:
            raise AppError("idle_video_unavailable", str(exc), 503) from exc
    return FileResponse(path, media_type="video/mp4")


@admin_router.get("/avatars/{avatar_id}/normalized.png")
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


@admin_router.post("/tts")
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


@admin_router.get("/jobs")
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


@admin_router.get("/sessions")
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
        out.append(
            {
                "session_id": row.id,
                "avatar_id": row.avatar_id,
                "status": render.state.value if render else row.status,
                "metrics": render.metrics if render else (row.metrics_json or {}),
            }
        )
    return out


@admin_router.post("/sessions/sweep")
def admin_sweep(db=Depends(get_db)) -> dict:
    import asyncio

    from server.main import get_session_manager

    manager = get_session_manager()
    closed = asyncio.run(manager.sweep_idle(db))
    return {"closed": closed}
