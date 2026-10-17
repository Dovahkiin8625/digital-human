"""End-to-end smoke: avatar upload -> ready -> session -> WS audio -> metrics -> close.

Exercises the whole committed chain through the public surfaces (admin REST,
session REST, audio WebSocket) using TestClient, so it is the one test that fails
if any two layers stop agreeing with each other.
"""

import io

import numpy as np
from PIL import Image


def test_full_flow_with_ws_audio(client):
    buf = io.BytesIO()
    Image.new("RGB", (800, 900), (90, 70, 60)).save(buf, format="PNG")
    avatar = client.post(
        "/api/v1/avatars",
        data={"name": "Alice"},
        files={"file": ("alice.png", buf.getvalue(), "image/png")},
    ).json()
    # The preprocessing job is a FastAPI BackgroundTask; TestClient runs it
    # synchronously before the response is returned, so the avatar is already
    # ready by the next request.
    assert client.get(f"/api/v1/avatars/{avatar['avatar_id']}").json()["status"] == "ready"

    session = client.post(
        "/api/v1/sessions", json={"avatar_id": avatar["avatar_id"]}
    ).json()
    sid = session["session_id"]
    assert session["signal_url"] == f"/api/v1/ws/signal/{sid}"

    with client.websocket_connect(f"/api/v1/ws/audio/{sid}") as ws:
        ws.send_bytes(np.zeros(48000, dtype=np.int16).tobytes())  # 1s of s16le mono 48kHz
        ws.send_text('{"type": "flush"}')

    detail = client.get(f"/api/v1/sessions/{sid}").json()
    # 48000 samples is 1.0s. AudioIngest splits into 100ms blocks (4800 samples
    # each) -> 10 blocks. MockEngine.drive computes n_frames = max(1, round(duration * fps));
    # per block duration == 0.1s and 0.1 * 25 == 2.5, which round() sends to 2
    # (banker's rounding) -> 2 frames/block * 10 blocks == 20 frames.
    assert detail["metrics"]["frames_out"] == 20

    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    closed = client.get(f"/api/v1/sessions/{sid}").json()
    assert closed["status"] == "closed"