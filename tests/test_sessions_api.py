import asyncio
import io

import numpy as np
from PIL import Image


def _seed(client):
    from sqlalchemy.orm import sessionmaker

    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    headers = {"X-API-Key": "key-1"}
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars", headers=headers, data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    return headers, resp.json()["avatar_id"]


def test_session_lifecycle(client):
    headers, avatar_id = _seed(client)
    resp = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id})
    assert resp.status_code == 201
    body = resp.json()
    assert body["signal_url"].endswith(body["session_id"])
    sid = body["session_id"]

    detail = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()
    assert detail["status"] in ("idle", "listening")
    assert detail["metrics"]["frames_out"] == 0

    assert client.delete(f"/api/v1/sessions/{sid}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}", headers=headers).json()["status"] == "closed"


def test_session_requires_ready_avatar(client):
    headers, _ = _seed(client)
    resp = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": "missing"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "avatar_not_ready"


def test_session_audio_drives_metrics(client):
    headers, avatar_id = _seed(client)
    sid = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}).json()["session_id"]
    render = client.app.state.session_manager.get_render(sid)
    asyncio.run(render.push_pcm(np.zeros(24000, dtype=np.int16).tobytes()))
    detail = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()
    # 24000 samples @48kHz = 0.5s -> 5x100ms ingest blocks; MockEngine emits
    # max(1, round(0.1*25))=2 frames/block = 10 (same math as test_render_session).
    assert detail["metrics"]["frames_out"] == 10


def test_sessions_list(client):
    headers, avatar_id = _seed(client)
    sid = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}).json()["session_id"]
    listing = client.get("/api/v1/sessions", headers=headers)
    assert listing.status_code == 200
    body = listing.json()
    assert [s["session_id"] for s in body] == [sid]
    assert body[0]["status"] == "listening"
    assert body[0]["metrics"]["frames_out"] == 0


def test_sweep_idle_closes_stale_sessions(client):
    headers, avatar_id = _seed(client)
    sid = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}).json()["session_id"]
    manager = client.app.state.session_manager
    render = manager.get_render(sid)

    from sqlalchemy.orm import sessionmaker

    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        assert asyncio.run(manager.sweep_idle(db)) == 0  # fresh session survives
        closed = asyncio.run(manager.sweep_idle(db, now=render.last_audio_at + manager.idle_timeout + 1))
        assert closed == 1
    finally:
        db.close()
    detail = client.get(f"/api/v1/sessions/{sid}", headers=headers).json()
    assert detail["status"] == "closed"


def test_attach_sink_receives_frames(client):
    from server.core.stream_sink import FrameBufferSink

    headers, avatar_id = _seed(client)
    sid = client.post("/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}).json()["session_id"]
    sink = FrameBufferSink()
    client.app.state.session_manager.attach_sink(sid, sink)
    render = client.app.state.session_manager.get_render(sid)
    asyncio.run(render.push_pcm(np.zeros(4800, dtype=np.int16).tobytes()))  # 100ms -> 2 frames
    assert sink.latest is not None
    assert render.metrics["frames_out"] == 2
