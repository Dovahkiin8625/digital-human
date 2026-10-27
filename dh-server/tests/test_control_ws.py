import io
import json
import time

import pytest
from PIL import Image
from starlette.websockets import WebSocketDisconnect


def _seed_avatar(client):
    buf = io.BytesIO()
    Image.new("RGB", (720, 1280), (10, 10, 80)).save(buf, format="PNG")
    r = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    avatar_id = r.json()["avatar_id"]
    for _ in range(50):
        rr = client.get(f"/api/v1/avatars/{avatar_id}")
        if rr.json()["status"] == "ready":
            break
        time.sleep(0.05)
    return avatar_id


def _session(client, avatar_id):
    return client.post(
        "/api/v1/sessions", json={"avatar_id": avatar_id}
    ).json()["session_id"]


def test_hello_sent_on_connect(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert "smile" in hello["vocab"]["expression"]
        assert hello["default_duration_ms"] == 1500


def test_trigger_expression_returns_ack_with_pts(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        ws.receive_json()  # hello
        ws.send_text(json.dumps({
            "type": "trigger",
            "event_id": "e1",
            "event": {"kind": "expression", "value": "smile", "duration_ms": 500},
        }))
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        assert ack["event_id"] == "e1"
        assert ack["applied_duration_ms"] == 500
        assert isinstance(ack["applied_at_pts"], (int, float))
        assert ack["applied_at_pts"] >= 0.0


def test_reset_does_not_disconnect(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "reset", "target": "expression"}))
        ws.send_text(json.dumps({"type": "ping"}))
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_unknown_token_disconnects(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({
            "type": "trigger",
            "event_id": "e2",
            "event": {"kind": "expression", "value": "happy", "duration_ms": 1000},
        }))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "unknown_token"
        # Server closes the socket after sending the error frame; the next
        # receive surfaces the WebSocketDisconnect.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_second_connection_rejected(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws1:
        ws1.receive_json()
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws2:
                ws2.receive_json()
                # Server should reject the second connection outright.


def test_session_delete_propagates_to_control_ws(client):
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        ws.receive_json()
        client.delete(f"/api/v1/sessions/{sid}")
        # Send a ping to unblock its receive loop; the handler then sees the
        # session has been removed and emits an error frame + close frame.
        ws.send_text(json.dumps({"type": "ping"}))
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "session_closed"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ack_clamps_overlong_duration_to_max(client):
    """Spec §5: ack carries applied (clamped) duration, not requested."""
    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({
            "type": "trigger",
            "event_id": "e_long",
            "event": {"kind": "expression", "value": "smile", "duration_ms": 999999},
        }))
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        assert ack["applied_duration_ms"] == 30000  # the cap, not 999999


def test_second_connection_close_code(client):
    """Review #12: when the second WS connects, the first must keep working
    and the second must be rejected with close 4001."""
    from starlette.websockets import WebSocketState

    avatar_id = _seed_avatar(client)
    sid = _session(client, avatar_id)
    with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws1:
        ws1.receive_json()
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"/api/v1/ws/control/{sid}") as ws2:
                ws2.receive_json()
        assert exc.value.code == 4001
        # First connection must keep working after the rejected second.
        ws1.send_text(json.dumps({"type": "ping"}))
        pong = ws1.receive_json()
        assert pong["type"] == "pong"