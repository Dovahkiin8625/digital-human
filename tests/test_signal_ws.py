import asyncio
import io
import json

import pytest
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from server.api.deps import hash_api_key


def _offer(*kinds):
    """Build a client offer with (kind, direction) transceivers; SDP only."""

    async def build():
        from aiortc import RTCPeerConnection

        pc = RTCPeerConnection()
        for kind, direction in kinds:
            pc.addTransceiver(kind, direction=direction)
        try:
            return (await pc.createOffer()).sdp
        finally:
            await pc.close()

    return asyncio.run(build())


def _mline_section(sdp: str, kind: str) -> str:
    """Return one m= block (lines from its m= up to the next m=)."""
    lines = sdp.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"m={kind} "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("m=")), len(lines))
    return "\n".join(lines[start:end])


def _mline_direction(sdp: str, kind: str) -> str | None:
    for line in _mline_section(sdp, kind).splitlines():
        if line in ("a=sendonly", "a=recvonly", "a=sendrecv"):
            return line[2:]
    return None


def _msid(sdp: str, kind: str) -> str:
    """The m-line's WMS stream id (first token of a=msid)."""
    for line in _mline_section(sdp, kind).splitlines():
        if line.startswith("a=msid:"):
            return line.split(":", 1)[1].split(" ", 1)[0]
    raise AssertionError(f"no a=msid in {kind} m-line")


def _cname(sdp: str, kind: str) -> str:
    """The SSRC-level cname the browser uses to align A/V RTCP timelines."""
    for line in _mline_section(sdp, kind).splitlines():
        if " cname:" in line:
            return line.split("cname:", 1)[1].strip()
    raise AssertionError(f"no cname in {kind} m-line")


def _bundle_mids(sdp: str) -> list[str]:
    for line in sdp.splitlines():
        if line.startswith("a=group:BUNDLE "):
            return line.split(" ", 1)[1].split(" ")
    raise AssertionError("no BUNDLE group in SDP")


def _seed(client):
    """Create a tenant + a ready avatar; return (api_key, avatar_id)."""
    from sqlalchemy.orm import sessionmaker

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
    return "key-1", resp.json()["avatar_id"]


def _new_session(client, api_key, avatar_id):
    return client.post(
        "/api/v1/sessions", headers={"X-API-Key": api_key}, json={"avatar_id": avatar_id}
    ).json()["session_id"]


def _add_tenant(name, api_key):
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Tenant
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name=name, api_key_hash=hash_api_key(api_key)))
    db.commit()
    db.close()


def test_signal_ws_rejects_invalid_api_key(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/v1/ws/signal/whatever?api_key=nope"):
            pass
    assert exc.value.code == 4401


def test_signal_ws_unknown_session_closes_4404(client):
    api_key, _ = _seed(client)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/api/v1/ws/signal/missing?api_key={api_key}") as ws:
            ws.receive_text()
    assert exc.value.code == 4404


def test_signal_ws_bye_restores_null_sink(client):
    from server.core.stream_sink import NullSink

    api_key, avatar_id = _seed(client)
    sid = _new_session(client, api_key, avatar_id)
    manager = client.app.state.session_manager

    with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key={api_key}") as ws:
        ws.send_text('{"type": "bye"}')
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    # Teardown restored the session's sink so audio ingest no longer feeds a dead track.
    assert isinstance(manager.get_render(sid).sink, NullSink)


def test_signal_ws_owner_key_attaches_sink(client):
    from server.core.webrtc import WebRTCSink

    api_key, avatar_id = _seed(client)
    sid = _new_session(client, api_key, avatar_id)
    manager = client.app.state.session_manager

    with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key={api_key}") as ws:
        # The owner's valid key attaches the live video sink.
        assert isinstance(manager.get_render(sid).sink, WebRTCSink)
        ws.send_text('{"type": "bye"}')
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_recvonly_offer_enables_server_audio_return(client):
    # The console offers audio recvonly: the answer must send audio back on the
    # same peer connection as video, and the sink must start feeding its track.
    from server.core.webrtc import WebRTCSink

    api_key, avatar_id = _seed(client)
    sid = _new_session(client, api_key, avatar_id)
    offer = _offer(("video", "recvonly"), ("audio", "recvonly"))
    manager = client.app.state.session_manager

    with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key={api_key}") as ws:
        ws.send_text(json.dumps({"type": "offer", "sdp": offer}))
        answer = json.loads(ws.receive_text())["sdp"]
        sink = manager.get_render(sid).sink

        assert _mline_direction(answer, "video") == "sendonly"
        assert _mline_direction(answer, "audio") == "sendonly"
        # One MediaStream (shared msid) over one BUNDLE with a shared cname: the
        # three SDP facts the browser uses to play audio and video in sync.
        assert _msid(answer, "audio") == _msid(answer, "video")
        assert _cname(answer, "audio") == _cname(answer, "video")
        assert set(_bundle_mids(answer)) == {
            ln.split(":", 1)[1].split(" ", 1)[0]
            for ln in answer.splitlines()
            if ln.startswith("a=mid:")
        }
        assert isinstance(sink, WebRTCSink)
        assert sink.wants_audio is True

        ws.send_text('{"type": "bye"}')
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_legacy_sendonly_offer_disables_audio_return(client):
    # Backwards compatibility: an older console only offers to SEND audio (the
    # server has nothing to play back on it). The answer stays recvonly and,
    # with no consumer draining the audio track, the sink must not feed it.
    api_key, avatar_id = _seed(client)
    sid = _new_session(client, api_key, avatar_id)
    offer = _offer(("video", "recvonly"), ("audio", "sendonly"))
    manager = client.app.state.session_manager

    with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key={api_key}") as ws:
        ws.send_text(json.dumps({"type": "offer", "sdp": offer}))
        answer = json.loads(ws.receive_text())["sdp"]
        sink = manager.get_render(sid).sink

        assert _mline_direction(answer, "audio") == "recvonly"
        assert sink.wants_audio is False

        ws.send_text('{"type": "bye"}')
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_signal_ws_rejects_other_tenants_session(client):
    from server.core.stream_sink import NullSink

    api_key, avatar_id = _seed(client)  # tenant A (key-1)
    _add_tenant("b", "key-2")           # tenant B, valid but not the owner
    sid = _new_session(client, api_key, avatar_id)
    manager = client.app.state.session_manager

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key=key-2") as ws:
            ws.receive_text()
    assert exc.value.code == 4404
    # B's rejected handshake must not have attached a video sink to A's session.
    assert isinstance(manager.get_render(sid).sink, NullSink)


def test_signal_ws_teardown_closes_pc_when_session_vanished(client, monkeypatch):
    """If the session is swept/deleted mid-connection, teardown must still close the pc."""
    from server.core import webrtc

    instances = []

    class _StubPC:
        def __init__(self):
            self.closed = False
            instances.append(self)

        def addTrack(self, track):
            pass

        def on(self, event, handler=None):
            # Real RTCPeerConnection.on supports both pc.on(event, handler) and
            # the @pc.on(event) decorator form; mirror it so the new "track"
            # registration in handle_signal_websocket is exercised.
            def decorator(fn):
                return fn

            return decorator if handler is None else handler

        async def close(self):
            self.closed = True

    monkeypatch.setattr(webrtc, "RTCPeerConnection", _StubPC)

    api_key, avatar_id = _seed(client)
    sid = _new_session(client, api_key, avatar_id)
    headers = {"X-API-Key": api_key}

    with client.websocket_connect(f"/api/v1/ws/signal/{sid}?api_key={api_key}") as ws:
        # The idle sweeper (or DELETE) removes the render while signaling is live.
        assert client.delete(f"/api/v1/sessions/{sid}", headers=headers).status_code == 204
        ws.send_text('{"type": "bye"}')
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    # A missing session must not abort teardown: the peer connection is released.
    assert instances and all(pc.closed for pc in instances)


def test_signal_ws_teardown_survives_client_disconnect(monkeypatch):
    """A client that disconnects first must not turn teardown into an exception.

    When the browser closes before the server does, Starlette's ``close()`` raises
    ``WebSocketDisconnect`` (1006); the handler must swallow it and return normally
    instead of letting uvicorn log an ASGI traceback on every ordinary disconnect.
    """
    import asyncio

    from server.core import webrtc
    from server.core.stream_sink import NullSink

    class _StubPC:
        def addTrack(self, track):
            pass

        def on(self, event, handler=None):
            def decorator(fn):
                return fn

            return decorator if handler is None else handler

        async def close(self):
            pass

    class _StubManager:
        def __init__(self):
            self.sinks = []

        def get_render(self, session_id):
            return object()

        def attach_sink(self, session_id, sink):
            self.sinks.append(sink)

    class _DisconnectingWS:
        async def accept(self):
            pass

        async def receive_text(self):
            return '{"type": "bye"}'

        async def close(self, code=1000):
            raise WebSocketDisconnect(code=1006)

    monkeypatch.setattr(webrtc, "RTCPeerConnection", _StubPC)

    manager = _StubManager()
    # Must return cleanly (not raise) even though close() reports the disconnect.
    asyncio.run(webrtc.handle_signal_websocket(_DisconnectingWS(), manager, "sid"))

    # Teardown still restored the null sink before the (guarded) close.
    assert isinstance(manager.sinks[-1], NullSink)

