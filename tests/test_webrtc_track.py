import numpy as np
import pytest

from server.core.stream_sink import SinkClosed
from server.core.webrtc import EngineVideoTrack, WebRTCSink


@pytest.mark.asyncio
async def test_track_push_recv_roundtrip():
    track = EngineVideoTrack()
    img = np.zeros((1280, 720, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    await track.push(img, 0.0)
    frame = await track.recv()
    assert frame.to_ndarray(format="rgb24")[0, 0, 0] == 255


@pytest.mark.asyncio
async def test_track_overflow_drops_oldest():
    track = EngineVideoTrack(maxsize=2)
    for i in range(4):
        await track.push(np.full((8, 8, 3), i, dtype=np.uint8), float(i))
    assert track.overflows >= 2
    frame = await track.recv()
    assert frame.to_ndarray(format="rgb24")[0, 0, 0] == 3


@pytest.mark.asyncio
async def test_sink_raises_after_close():
    track = EngineVideoTrack()
    sink = WebRTCSink(track)
    await sink.close()
    with pytest.raises(SinkClosed):
        await sink.push(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
