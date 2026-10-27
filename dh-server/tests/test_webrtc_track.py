import asyncio

import numpy as np
import pytest

from server.core.stream_sink import SinkClosed
from server.core.webrtc import EngineAudioTrack, EngineVideoTrack, WebRTCSink


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
    # Overflow drops the oldest on push; recv() hands out what survived, oldest
    # first, so a consumer that keeps up sees every frame in order.
    assert (await track.recv()).to_ndarray(format="rgb24")[0, 0, 0] == 2
    assert (await track.recv()).to_ndarray(format="rgb24")[0, 0, 0] == 3


@pytest.mark.asyncio
async def test_sink_raises_after_close():
    track = EngineVideoTrack()
    sink = WebRTCSink(track, EngineAudioTrack())
    await sink.close()
    with pytest.raises(SinkClosed):
        await sink.push(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)


# --- audio return track -----------------------------------------------------


@pytest.mark.asyncio
async def test_audio_track_push_recv_roundtrip():
    track = EngineAudioTrack()
    pcm = np.arange(960, dtype=np.int16)
    await track.push(pcm, 0)
    frame = await track.recv()
    assert frame.sample_rate == 48000
    assert frame.layout.name == "mono"
    assert frame.format.name == "s16"
    assert frame.pts == 0
    assert frame.to_ndarray().shape == (1, 960)


@pytest.mark.asyncio
async def test_audio_track_overflow_drops_oldest():
    track = EngineAudioTrack(maxsize=2)
    for pts in range(4):
        await track.push(np.full(960, pts, dtype=np.int16), pts)
    assert track.overflows >= 2
    assert (await track.recv()).pts == 2
    assert (await track.recv()).pts == 3


@pytest.mark.asyncio
async def test_sink_push_audio_splits_100ms_block_into_20ms_frames():
    audio = EngineAudioTrack()
    sink = WebRTCSink(EngineVideoTrack(), audio)
    # The block RenderSession hands the engine: float32 mono, 100 ms @ 48 kHz.
    await sink.push_audio(np.full(4800, 0.25, dtype=np.float32), 0.0, 48000)
    frames = [await audio.recv() for _ in range(5)]
    assert [f.pts for f in frames] == [0, 960, 1920, 2880, 3840]
    for f in frames:
        assert f.sample_rate == 48000
        assert f.to_ndarray().shape == (1, 960)
    # float->s16 is truncation, same as the RTP ingest path: 0.25 * 32767 = 8191.75 -> 8191
    assert frames[0].to_ndarray()[0, 0] == 8191


@pytest.mark.asyncio
async def test_sink_push_audio_pads_partial_tail_and_keeps_pts():
    audio = EngineAudioTrack()
    sink = WebRTCSink(EngineVideoTrack(), audio)
    await sink.push_audio(np.full(100, 1.0, dtype=np.float32), 0.45, 48000)
    frame = await audio.recv()
    pcm = frame.to_ndarray()[0]
    assert frame.pts == 21600  # 0.45 s * 48000
    assert pcm[:100].tolist() == [32767] * 100
    assert pcm[100:].tolist() == [0] * 860  # padded with silence
    # The padded tail is delivered now, not held for a future block.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(audio.recv(), timeout=0.05)


@pytest.mark.asyncio
async def test_sink_push_audio_dropped_when_not_negotiated():
    audio = EngineAudioTrack()
    sink = WebRTCSink(EngineVideoTrack(), audio)
    sink.wants_audio = False
    await sink.push_audio(np.zeros(4800, dtype=np.float32), 0.0, 48000)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(audio.recv(), timeout=0.05)


@pytest.mark.asyncio
async def test_sink_push_audio_raises_after_close():
    sink = WebRTCSink(EngineVideoTrack(), EngineAudioTrack())
    await sink.close()
    with pytest.raises(SinkClosed):
        await sink.push_audio(np.zeros(4800, dtype=np.float32), 0.0, 48000)
