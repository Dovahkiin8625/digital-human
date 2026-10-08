import hmac
import io
import json
from pathlib import Path
from secrets import token_urlsafe
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select
from starlette.responses import Response

from server.api.deps import get_admin, hash_api_key
from server.avatar.pipeline import run_job_in_new_session
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


@router.post("/login")
def login(body: LoginIn) -> dict:
    expected = get_settings().admin_password
    if not hmac.compare_digest(body.password.encode("utf-8"), expected.encode("utf-8")):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
    return {"token": get_settings().admin_token}


@admin_router.get("/system")
def system_info() -> dict:
    from server.engine.base import create_engine

    engine = create_engine()
    described = engine.describe()
    return {
        "engine": described,
        "version": "0.1.0",
        "data_dir": str(get_settings().data_dir),
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


@admin_router.get("/avatars")
def admin_list_avatars(tenant_id: str = "", db=Depends(get_db)) -> list[dict]:
    stmt = select(Avatar)
    if tenant_id:
        stmt = stmt.where(Avatar.tenant_id == tenant_id)
    rows = db.execute(stmt.order_by(Avatar.created_at.desc())).scalars().all()
    out = []
    for a in rows:
        job = (
            db.execute(
                select(AvatarJob)
                .where(AvatarJob.avatar_id == a.id)
                .order_by(AvatarJob.created_at.desc())
            )
            .scalars()
            .first()
        )
        out.append(
            {
                "avatar_id": a.id,
                "tenant_id": a.tenant_id,
                "name": a.name,
                "status": a.status,
                "config": a.config_json,
                "error": job.error if a.status == "failed" and job else None,
                "created_at": a.created_at.isoformat(),
            }
        )
    return out


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


@admin_router.get("/avatars/{avatar_id}/preview.png")
def avatar_preview(avatar_id: str, level: float = Query(0.5, ge=0.0, le=1.0), db=Depends(get_db)):
    import asyncio

    from server.engine.base import create_engine as _ce

    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    engine = _ce()
    audio = np.ones(4800, dtype=np.float32) * level
    frames = asyncio.run(engine.drive(pack, audio, 48000))
    buf = io.BytesIO()
    Image.fromarray(frames[0].image).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


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
