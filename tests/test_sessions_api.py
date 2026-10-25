import asyncio
import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from doubles import FakeEngine


def _seed(client):
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars", data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    return resp.json()["avatar_id"]


def _new_session(client, avatar_id):
    return client.post("/api/v1/sessions", json={"avatar_id": avatar_id}).json()["session_id"]


def test_session_lifecycle(client):
    avatar_id = _seed(client)
    resp = client.post("/api/v1/sessions", json={"avatar_id": avatar_id})
    assert resp.status_code == 201
    body = resp.json()
    assert body["signal_url"].endswith(body["session_id"])
    sid = body["session_id"]

    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["status"] in ("idle", "listening")
    assert detail["metrics"]["frames_out"] == 0

    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").json()["status"] == "closed"


def test_session_requires_ready_avatar(client):
    _seed(client)
    resp = client.post("/api/v1/sessions", json={"avatar_id": "missing"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "avatar_not_ready"


def test_session_audio_drives_metrics(client):
    avatar_id = _seed(client)
    sid = _new_session(client, avatar_id)
    render = client.app.state.session_manager.get_render(sid)
    asyncio.run(render.push_pcm(np.zeros(24000, dtype=np.int16).tobytes()))
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    # 24000 samples @48kHz = 0.5s -> 5x100ms ingest blocks; FakeEngine emits
    # max(1, round(0.1*25))=2 frames/block = 10 (same math as test_render_session).
    assert detail["metrics"]["frames_out"] == 10


def test_sessions_list(client):
    avatar_id = _seed(client)
    sid = _new_session(client, avatar_id)
    listing = client.get("/api/v1/sessions")
    assert listing.status_code == 200
    body = listing.json()
    assert [s["session_id"] for s in body] == [sid]
    assert body[0]["status"] == "listening"
    assert body[0]["metrics"]["frames_out"] == 0


def test_sweep_idle_closes_stale_sessions(client):
    avatar_id = _seed(client)
    sid = _new_session(client, avatar_id)
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
    detail = client.get(f"/api/v1/sessions/{sid}").json()
    assert detail["status"] == "closed"


def test_attach_sink_receives_frames(client):
    from server.core.stream_sink import FrameBufferSink

    avatar_id = _seed(client)
    sid = _new_session(client, avatar_id)
    sink = FrameBufferSink()
    client.app.state.session_manager.attach_sink(sid, sink)
    render = client.app.state.session_manager.get_render(sid)
    asyncio.run(render.push_pcm(np.zeros(4800, dtype=np.int16).tobytes()))  # 100ms -> 2 frames
    assert sink.latest is not None
    assert render.metrics["frames_out"] == 2


class _CloseTrackingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def test_session_close_does_not_close_engine_but_shutdown_does(tmp_path, monkeypatch):
    from server.avatar.store import AvatarStore
    from server.core.render_session import RenderSession
    from server.core.session_manager import SessionManager
    from server.core.stream_sink import NullSink
    from server.engine.base import AvatarPack
    from server.engine.pool import EnginePool

    img = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img, config={})
    engine = _CloseTrackingEngine()
    monkeypatch.setattr("server.engine.base.create_engine", lambda *a, **k: engine)

    # A RenderSession close must NOT tear down the shared engine.
    render = RenderSession("s1", pack, engine, NullSink())
    asyncio.run(render.start())
    asyncio.run(render.close())
    assert engine.close_calls == 0

    # The pool owns the engines (one per digital-human model); shutdown
    # closes them exactly once (idempotent).
    pool = EnginePool()
    assert pool.engine_for("musetalk") is engine
    manager = SessionManager(pool, AvatarStore(tmp_path), idle_timeout=60.0)
    asyncio.run(manager.shutdown())
    asyncio.run(manager.shutdown())
    assert engine.close_calls == 1


def _build_app(tmp_path, monkeypatch, **env):
    from doubles import install_doubles
    from env import apply_env

    apply_env(monkeypatch, tmp_path, **env)
    # Same swap the ``client`` fixture does: the app's engine and pipeline
    # are the doubles, not the GPU-backed MuseTalk model.
    install_doubles(monkeypatch)
    from server.db.session import get_engine, init_db, reset_engine
    from server.main import create_app

    reset_engine()
    app = create_app()
    init_db(get_engine())
    return app


def test_session_honors_configured_fps_and_sample_rate(tmp_path, monkeypatch):
    from server.config import get_settings
    from server.db.session import reset_engine

    app = _build_app(tmp_path, monkeypatch, DH_FPS="30", DH_SAMPLE_RATE="16000")
    try:
        with TestClient(app) as c:
            avatar_id = _seed(c)
            sid = c.post(
                "/api/v1/sessions", json={"avatar_id": avatar_id}
            ).json()["session_id"]
            render = app.state.session_manager.get_render(sid)
            # The render must carry the configured rates, not the hardcoded 25/48000,
            # or engine fps and pts would silently drift apart.
            assert render.fps == 30
            assert render.sample_rate == 16000
            # 100ms ingest blocks at 16kHz are 1600 samples.
            assert render._ingest.block_samples == 1600
    finally:
        reset_engine()
        get_settings.cache_clear()


def test_lifespan_sweeper_reclaims_idle_and_stops_cleanly(tmp_path, monkeypatch):
    app = _build_app(
        tmp_path, monkeypatch,
        DH_SESSION_IDLE_TIMEOUT="0.05", DH_SESSION_SWEEP_INTERVAL="0.02",
    )
    from server.config import get_settings
    from server.db.session import reset_engine

    try:
        with TestClient(app) as c:
            task = app.state.sweep_task
            assert task is not None and not task.done()

            avatar_id = _seed(c)
            sid = c.post(
                "/api/v1/sessions", json={"avatar_id": avatar_id}
            ).json()["session_id"]

            status = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                status = c.get(f"/api/v1/sessions/{sid}").json()["status"]
                if status == "closed":
                    break
                time.sleep(0.02)
            assert status == "closed", f"background sweeper never reclaimed idle session ({status})"
        # Leaving the context ran lifespan shutdown: the sweep task was cancelled/awaited.
        assert task.done()
    finally:
        reset_engine()
        get_settings.cache_clear()


def test_sweep_loop_survives_transient_failure(tmp_path, monkeypatch):
    app = _build_app(
        tmp_path, monkeypatch,
        DH_SESSION_IDLE_TIMEOUT="0.05", DH_SESSION_SWEEP_INTERVAL="0.02",
    )
    from server.config import get_settings
    from server.db.session import reset_engine

    try:
        with TestClient(app) as c:
            manager = app.state.session_manager
            calls = {"n": 0}
            real = manager.sweep_idle

            async def flaky(db, now=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient sweep failure")
                return await real(db, now)

            monkeypatch.setattr(manager, "sweep_idle", flaky)

            deadline = time.time() + 5.0
            while calls["n"] < 2 and time.time() < deadline:
                time.sleep(0.02)

            # One bad tick must not kill the sweeper: it keeps running and recovers.
            assert calls["n"] >= 2, "sweep loop died after a transient tick failure"
            assert not app.state.sweep_task.done()
    finally:
        reset_engine()
        get_settings.cache_clear()