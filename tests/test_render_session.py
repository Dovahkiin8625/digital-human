import numpy as np
import pytest

from server.core.audio_ingest import AudioIngest
from server.core.render_session import RenderSession, SessionState
from server.core.stream_sink import FrameBufferSink
from server.engine.base import AvatarPack
from server.engine.mock_engine import MockEngine


def test_audio_ingest_blocks():
    ingest = AudioIngest(sample_rate=48000, block_ms=100)
    pcm = np.zeros(4800, dtype=np.int16).tobytes()  # 100ms
    blocks = ingest.push_pcm(pcm)
    assert len(blocks) == 1
    assert blocks[0].shape == (4800,)
    assert blocks[0].dtype == np.float32
    assert ingest.push_pcm(np.zeros(2400, dtype=np.int16).tobytes()) == []
    flushed = ingest.flush()
    assert len(flushed) == 1 and flushed[0].shape == (2400,)


def _session(tmp_path):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (10, 10, 10)).save(img)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img, config={})
    sink = FrameBufferSink()
    return RenderSession("s1", pack, MockEngine(), sink), sink


@pytest.mark.asyncio
async def test_push_pcm_drives_frames_and_metrics(tmp_path):
    session, sink = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 0.5s → 5×100ms blocks
    assert session.state == SessionState.SPEAKING
    # 5 blocks × round(0.1s*25fps)=round(2.5)=2 frames/block = 10 (MockEngine math)
    assert session.metrics["frames_out"] == 10
    assert sink.latest is not None
    assert sink.latest[1].shape == (1280, 720, 3)
    await session.close()
    assert session.state == SessionState.CLOSED


@pytest.mark.asyncio
async def test_push_after_close_raises(tmp_path):
    from server.core.errors import AppError

    session, _ = _session(tmp_path)
    await session.start()
    await session.close()
    with pytest.raises(AppError) as exc:
        await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    assert exc.value.code == "session_closed"


@pytest.mark.asyncio
async def test_maybe_idle_transitions(tmp_path):
    session, _ = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.LISTENING
    await session.close()
