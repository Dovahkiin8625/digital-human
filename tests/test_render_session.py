import asyncio

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


def test_audio_ingest_flush_odd_residual():
    ingest = AudioIngest(sample_rate=48000, block_ms=100)
    ingest.push_pcm(b"\x00" * 201)  # 100 whole samples + 1 stray byte
    flushed = ingest.flush()
    assert len(flushed) == 1 and flushed[0].shape == (100,)
    assert ingest.flush() == []  # stray byte dropped, not retained

    lone = AudioIngest(sample_rate=48000, block_ms=100)
    lone.push_pcm(b"\x7f")  # half a sample only
    assert lone.flush() == []


class _RecordingSink(FrameBufferSink):
    def __init__(self):
        super().__init__()
        self.pts_list: list[float] = []

    async def push(self, image: np.ndarray, pts: float) -> None:
        self.pts_list.append(pts)
        await super().push(image, pts)


class _BoomEngine(MockEngine):
    async def drive(self, pack, audio, sample_rate, *, context=None):
        raise RuntimeError("engine exploded")


def _session(tmp_path, engine=None, sink=None):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (10, 10, 10)).save(img)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img, config={})
    sink = sink if sink is not None else FrameBufferSink()
    return RenderSession("s1", pack, engine or MockEngine(), sink), sink


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
async def test_pts_tracks_audio_time(tmp_path):
    sink = _RecordingSink()
    session, _ = _session(tmp_path, sink=sink)
    await session.start()
    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 0.5s → 10 frames
    pts = sink.pts_list
    assert len(pts) == 10
    assert pts[0] == pytest.approx(0.0)
    # frames distributed across each 100ms block: last pts = 0.4 + 0.1/2
    assert pts[-1] == pytest.approx(0.45)
    assert pts[-1] < 0.5
    assert all(b > a for a, b in zip(pts, pts[1:]))  # monotonic
    await session.close()


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
async def test_engine_error_marks_session_and_reraises(tmp_path):
    session, _ = _session(tmp_path, engine=_BoomEngine())
    await session.start()
    with pytest.raises(RuntimeError, match="engine exploded"):
        await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    assert session.state == SessionState.ERROR
    assert session.error is not None and "engine exploded" in session.error


@pytest.mark.asyncio
async def test_maybe_idle_transitions(tmp_path):
    session, _ = _session(tmp_path)
    await session.start()
    assert session.state == SessionState.LISTENING
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    assert session.state == SessionState.SPEAKING
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.IDLE
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())  # IDLE → SPEAKING
    assert session.state == SessionState.SPEAKING
    await session.close()


@pytest.mark.asyncio
async def test_close_flushes_audio_tail(tmp_path):
    session, _ = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(2400, dtype=np.int16).tobytes())  # 50ms tail, no full block
    assert session.metrics["frames_out"] == 0
    await session.close()
    assert session.state == SessionState.CLOSED
    # tail driven on close: max(1, round(0.05 * 25)) = 1 frame
    assert session.metrics["frames_out"] == 1


class _GateEngine(MockEngine):
    """Yields inside the first drive() so a concurrent close() can interleave.

    MockEngine.drive never awaits, so it cannot expose the missing lock; this
    stand-in mimics a real (GPU) engine whose drive() suspends.
    """

    def __init__(self):
        super().__init__()
        self._first = True
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def drive(self, pack, audio, sample_rate, *, context=None):
        if self._first:
            self._first = False
            self.entered.set()
            await self.gate.wait()
        return await super().drive(pack, audio, sample_rate, context=context)


@pytest.mark.asyncio
async def test_push_concurrent_with_close_is_serialized(tmp_path):
    sink = _RecordingSink()
    engine = _GateEngine()
    session, _ = _session(tmp_path, engine=engine, sink=sink)
    await session.start()

    # 0.55s: five full 100ms blocks drain immediately, a 0.05s tail stays buffered.
    pcm = np.zeros(26400, dtype=np.int16).tobytes()
    push = asyncio.create_task(session.push_pcm(pcm))
    await engine.entered.wait()  # push is parked inside its first drive(), holding the lock
    close = asyncio.create_task(session.close())
    await asyncio.sleep(0)  # close must block on the lock rather than run concurrently
    await asyncio.sleep(0)
    engine.gate.set()
    await asyncio.gather(push, close)

    assert session.state == SessionState.CLOSED
    pts = sink.pts_list
    # Serialized: every frame lands in strictly increasing pts order and each is
    # counted once. Without the lock close() would drive the tail (pts 0.1) before
    # push's first block (pts 0.0), breaking monotonicity.
    assert all(b > a for a, b in zip(pts, pts[1:]))
    assert session.metrics["frames_out"] == len(pts)
