import json

from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy import select
from starlette.responses import Response

from server.api.deps import get_tenant, hash_api_key
from server.core.errors import AppError
from server.db.models import Avatar, StreamSession, Tenant
from server.db.session import get_db

router = APIRouter(tags=["sessions"])


def _manager():
    from server.main import get_session_manager

    return get_session_manager()


def _serialize(row: StreamSession, render=None) -> dict:
    data = {
        "session_id": row.id,
        "avatar_id": row.avatar_id,
        "status": row.status,
        "metrics": row.metrics_json or {},
    }
    if render is not None:
        data["status"] = render.state.value
        data["metrics"] = render.metrics
    return data


@router.post("/api/v1/sessions", status_code=201)
async def create_session(body: dict, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    avatar_id = body.get("avatar_id", "")
    avatar = db.execute(
        select(Avatar).where(Avatar.id == avatar_id, Avatar.tenant_id == tenant.id)
    ).scalar_one_or_none()
    if avatar is None or avatar.status != "ready":
        raise AppError("avatar_not_ready", f"avatar {avatar_id} is not ready", 409)
    row = await _manager().create(db, tenant.id, avatar)
    return {
        "session_id": row.id,
        "status": row.status,
        "signal_url": f"/api/v1/ws/signal/{row.id}",
        "audio_url": f"/api/v1/ws/audio/{row.id}",
    }


@router.get("/api/v1/sessions")
def list_sessions(tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> list[dict]:
    rows = (
        db.execute(
            select(StreamSession)
            .where(StreamSession.tenant_id == tenant.id)
            .order_by(StreamSession.started_at.desc())
        )
        .scalars()
        .all()
    )
    out = []
    for row in rows:
        render = None
        try:
            render = _manager().get_render(row.id)
        except AppError:
            pass
        out.append(_serialize(row, render))
    return out


@router.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> dict:
    row = db.execute(
        select(StreamSession).where(StreamSession.id == session_id, StreamSession.tenant_id == tenant.id)
    ).scalar_one_or_none()
    if row is None:
        raise AppError("not_found", f"session {session_id} not found", 404)
    try:
        render = _manager().get_render(session_id)
    except AppError:
        render = None
    return _serialize(row, render)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, tenant: Tenant = Depends(get_tenant), db=Depends(get_db)) -> Response:
    row = db.execute(
        select(StreamSession).where(StreamSession.id == session_id, StreamSession.tenant_id == tenant.id)
    ).scalar_one_or_none()
    if row is None:
        raise AppError("not_found", f"session {session_id} not found", 404)
    await _manager().close(db, session_id)
    return Response(status_code=204)


def _ws_auth(api_key: str) -> bool:
    """WebSocket auth: REST uses the X-API-Key header, WS uses ?api_key=."""
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Tenant
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        tenant = db.query(Tenant).filter(Tenant.api_key_hash == hash_api_key(api_key)).first()
        return tenant is not None
    finally:
        db.close()


@router.websocket("/api/v1/ws/signal/{session_id}")
async def signal_ws(websocket: WebSocket, session_id: str, api_key: str = "") -> None:
    if not _ws_auth(api_key):
        await websocket.close(code=4401)
        return
    from server.core.webrtc import handle_signal_websocket

    await handle_signal_websocket(websocket, _manager(), session_id)


@router.websocket("/api/v1/ws/audio/{session_id}")
async def audio_ws(websocket: WebSocket, session_id: str, api_key: str = "") -> None:
    """Backup PCM ingress: binary frames are raw s16le mono 48kHz chunks fed to
    the same RenderSession.push_pcm as the RTP audio track; a text {"type":"flush"}
    drains the ingest's partial tail block."""
    if not _ws_auth(api_key):
        await websocket.close(code=4401)
        return
    try:
        render = _manager().get_render(session_id)
    except AppError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes") is not None:
                await render.push_pcm(msg["bytes"])
            elif msg.get("text"):
                payload = json.loads(msg["text"])
                if payload.get("type") == "flush":
                    for block in render._ingest.flush():
                        await render._drive_block(block)
    except Exception:
        pass
