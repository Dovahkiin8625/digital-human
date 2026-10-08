import io

import av
import numpy as np
import pytest
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from server.core.webrtc import consume_audio_track


def _setup(client):
    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    headers = {"X-API-Key": "key-1"}
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (1, 2, 3)).save(buf, format="PNG")
    avatar_id = client.post(
        "/api/v1/avatars", headers=headers, data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]
    sid = client.post(
        "/api/v1/sessions", headers=headers, json={"avatar_id": avatar_id}
    ).json()["session_id"]
    return sid


def test_ws_audio_channel_drives_frames(client):
    sid = _setup(client)
    with client.websocket_connect(f"/api/v1/ws/audio/{sid}?api_key=key-1") as ws:
        ws.send_bytes(np.zeros(4800, dtype=np.int16).tobytes())  # 100ms → 1 ingest block
        ws.send_text('{"type": "flush"}')

    render = client.app.state.session_manager.get_render(sid)
    # MockEngine at 25fps emits round(0.1s * 25) = 2 frames for a 100ms block
    # (banker's rounding). The brief's "5 frames" is a math slip: 100ms of audio
    # is one 100ms ingest block, driven once. Matches Task 6's ruling on the
    # same 100ms-block arithmetic.
    assert render.metrics["frames_out"] == 2


def test_ws_audio_rejects_bad_key(client):
    sid = _setup(client)
    # Assert the required close code, not merely "some disconnect": an
    # auth-boundary test that swallows every exception proves nothing.
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/api/v1/ws/audio/{sid}?api_key=bad") as ws:
            ws.receive()
    assert exc.value.code == 4401


class _FakeTrack:
    """Minimal MediaStreamTrack stand-in: yields queued frames, then ends."""

    def __init__(self, frames):
        self._frames = list(frames)

    async def recv(self):
        if not self._frames:
            raise RuntimeError("track ended")
        return self._frames.pop(0)


def _audio_frame(samples, rate, layout, fmt="s16"):
    channels = 1 if layout == "mono" else 2
    arr = np.zeros((channels, samples), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(arr, format=fmt, layout=layout)
    frame.sample_rate = rate
    return frame


@pytest.mark.asyncio
async def test_consume_audio_track_passes_through_48k_mono():
    pushed = []

    async def push_pcm(data):
        pushed.append(data)

    track = _FakeTrack([_audio_frame(480, 48000, "mono")])
    await consume_audio_track(track, push_pcm)  # second recv() ends the loop

    assert len(pushed) == 1
    assert len(pushed[0]) == 480 * 2  # s16 mono → 2 bytes per sample


@pytest.mark.asyncio
async def test_consume_audio_track_exits_silently_on_error():
    async def push_pcm(data):  # pragma: no cover - must never be called
        raise AssertionError("push_pcm should not be reached")

    class _Boom:
        async def recv(self):
            raise RuntimeError("transport gone")

    # A dead track must not propagate; the consumer returns quietly.
    await consume_audio_track(_Boom(), push_pcm)
