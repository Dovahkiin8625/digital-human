import asyncio

import numpy as np
import pytest

from server.core.audio_ingest import AudioIngest
from server.core.render_session import RenderSession, SessionState
from server.core.render_session import _histogram
from server.core.stream_sink import FrameBufferSink, NullSink, SinkClosed
from server.engine.base import AvatarPack
from doubles import FakeEngine


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


class _AudioSink(_RecordingSink):
    """A live sink that also asks for audio return, like WebRTCSink."""

    keepalive = True
    wants_audio = True

    def __init__(self):
        super().__init__()
        self.audio: list[tuple[float, int, int]] = []
        self.audio_samples: list[np.ndarray] = []
        self.events: list[tuple[str, float]] = []

    async def push(self, image, pts):
        self.events.append(("v", pts))
        await super().push(image, pts)

    async def push_audio(self, samples, pts, sample_rate):
        self.audio.append((float(pts), len(samples), sample_rate))
        self.audio_samples.append(np.asarray(samples))
        self.events.append(("a", float(pts)))


class _BoomEngine(FakeEngine):
    async def drive(self, pack, audio, sample_rate, *, context=None):
        raise RuntimeError("engine exploded")


def _session(tmp_path, engine=None, sink=None):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (10, 10, 10)).save(img)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img, config={})
    sink = sink if sink is not None else FrameBufferSink()
    return RenderSession("s1", pack, engine or FakeEngine(), sink), sink


@pytest.mark.asyncio
async def test_push_pcm_drives_frames_and_metrics(tmp_path):
    session, sink = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 0.5s → 5×100ms blocks
    assert session.state == SessionState.SPEAKING
    # 5 blocks × round(0.1s*25fps)=round(2.5)=2 frames/block = 10 (FakeEngine math)
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
async def test_drive_block_forwards_its_audio_to_the_sink(tmp_path):
    sink = _AudioSink()
    session, _ = _session(tmp_path, sink=sink)
    await session.start()
    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 0.5 s, five blocks
    # Each 100 ms engine block is returned as audio at its own timeline origin.
    assert [p for p, _, _ in sink.audio] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert {n for _, n, _ in sink.audio} == {4800}
    assert {rate for _, _, rate in sink.audio} == {48000}
    # The engine receives the same float32 samples the listener's browser will.
    await session.close()


@pytest.mark.asyncio
async def test_block_audio_leaves_before_its_video_frames(tmp_path):
    sink = _AudioSink()
    session, _ = _session(tmp_path, sink=sink)
    await session.start()
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())  # one block
    # Audio packet must precede that block's video packets so the browser, which
    # treats audio as its master clock, never gets video ahead of the sound.
    assert sink.events[0] == ("a", 0.0)
    assert sink.events[1][0] == "v"
    await session.close()


@pytest.mark.asyncio
async def test_keepalive_fills_idle_audio_with_silence_on_matching_pts(tmp_path):
    sink = _AudioSink()
    session, _ = _session_with_canvas(tmp_path, sink=sink)
    await session.start()
    await asyncio.sleep(0.2)  # ~5 keep-alive ticks at 25 fps (40 ms each)
    audio_before = list(sink.audio)
    video_pts = list(sink.pts_list)
    await session.close()

    assert audio_before, "idle session must keep the audio timeline populated"
    # Every tick: one 40 ms silence buffer (1920 samples @ 48 kHz) ...
    assert all(n == 1920 and rate == 48000 for _, n, rate in audio_before)
    assert all(np.all(s == 0) for s in sink.audio_samples)  # ... actually silent
    # ... sharing exactly the video frames' pts -- no audio/video clock offset.
    assert [p for p, _, _ in audio_before] == pytest.approx(video_pts)


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
async def test_maybe_idle_returns_to_idle_within_one_second(tmp_path):
    """After speech ends, the session must reach IDLE within ~1s.

    Regression for the perceived ~1.7s freeze on the last speech frame after
    audio stopped: IDLE_AFTER was 2.0s, so the WebRTC video queue drained
    (~320 ms for maxsize=8 at 25 fps) and then the receiver sat on the last
    frame until the keep-alive loop was allowed to push idle frames again.
    With IDLE_AFTER at half a second the queue has barely started draining
    before idle frames take over, so the freeze is bounded by the queue.
    """
    session, _ = _session(tmp_path)
    await session.start()
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())
    assert session.state == SessionState.SPEAKING
    # 1.0s is past the new IDLE_AFTER (0.5s) but well under the old 2.0s --
    # so a regression to the old value flips this back to SPEAKING and fails.
    session.maybe_idle(now=session.last_audio_at + 1.0)
    assert session.state == SessionState.IDLE
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


class _GateEngine(FakeEngine):
    """Yields inside the first drive() so a concurrent close() can interleave.

    FakeEngine.drive never awaits, so it cannot expose the missing lock; this
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


class _LiveSink(_RecordingSink):
    """A sink that asks for idle keep-alive frames, as WebRTCSink does."""

    keepalive = True


def _session_with_canvas(tmp_path, sink=None):
    """Like `_session`, but with the resting canvas every pipeline writes."""
    from PIL import Image

    session, sink = _session(tmp_path, sink=sink if sink is not None else _LiveSink())
    Image.new("RGB", (720, 1280), (200, 30, 40)).save(tmp_path / "normalized.png")
    return session, sink


@pytest.mark.asyncio
async def test_idle_session_keeps_a_live_track_fed(tmp_path):
    """An idle session drives nothing, so its track would starve and freeze."""
    session, sink = _session_with_canvas(tmp_path)
    await session.start()
    await asyncio.sleep(0.2)
    # FrameBufferSink.close() drops its buffer, so read it before closing.
    assert sink.latest is not None
    assert tuple(sink.latest[1][0, 0]) == (200, 30, 40)
    assert session.metrics["keepalive_frames"] > 0
    # Nothing was driven, so the driving pipeline's own counter stays at zero.
    assert session.metrics["frames_out"] == 0
    await session.close()


@pytest.mark.asyncio
async def test_keepalive_advances_the_session_clock(tmp_path):
    """Repeated pts would read as one frame; a jump would break the resume."""
    session, sink = _session_with_canvas(tmp_path)
    await session.start()
    await asyncio.sleep(0.2)
    await session.close()

    assert len(sink.pts_list) > 1
    assert sink.pts_list == sorted(sink.pts_list)
    assert sink.pts_list[1] > sink.pts_list[0]


@pytest.mark.asyncio
async def test_keepalive_pauses_while_speaking(tmp_path):
    session, sink = _session_with_canvas(tmp_path)
    await session.start()
    await asyncio.sleep(0.15)
    await session.push_pcm(np.zeros(4800, dtype=np.int16).tobytes())  # -> SPEAKING
    at = session.metrics["keepalive_frames"]
    await asyncio.sleep(0.15)
    assert session.state == SessionState.SPEAKING
    assert session.metrics["keepalive_frames"] == at
    await session.close()


@pytest.mark.asyncio
async def test_null_sink_gets_no_keepalive_frames(tmp_path):
    """Nothing is attached, so there is no track to keep alive."""
    session, _ = _session_with_canvas(tmp_path, sink=NullSink())
    await session.start()
    await asyncio.sleep(0.1)
    await session.close()
    assert session.metrics["keepalive_frames"] == 0
    assert session.metrics["frames_out"] == 0


@pytest.mark.asyncio
async def test_missing_canvas_disables_keepalive(tmp_path):
    """A pack without normalized.png must not break the session."""
    session, _ = _session(tmp_path, sink=_LiveSink())
    await session.start()
    await asyncio.sleep(0.1)
    await session.close()
    assert session.metrics["keepalive_frames"] == 0


@pytest.mark.asyncio
async def test_keepalive_exits_on_error_even_without_a_canvas(tmp_path):
    """CLOSED/ERROR must stop the task before the keep-alive gate.

    The check used to live only inside the lock, which a NullSink -- or a pack
    with no normalized.png -- never reaches, so the task spun at 25 Hz forever
    after the session errored. `_session` here writes no canvas and uses a
    NullSink, which is exactly that blind spot.
    """
    session, _ = _session(tmp_path, sink=NullSink())
    await session.start()
    task = session._keepalive_task
    assert task is not None
    session.mark_error("boom")
    await asyncio.sleep(0.15)
    assert task.done()
    await session.close()


@pytest.mark.asyncio
async def test_stop_cancels_the_keepalive_task(tmp_path):
    session, _ = _session_with_canvas(tmp_path)
    await session.start()
    assert session._keepalive_task is not None
    session.stop()
    assert session._keepalive_task is None
    await asyncio.sleep(0)
    await session.close()


def test_latency_histogram_buckets():
    assert _histogram([]) == {}
    got = _histogram([10.0, 30.0, 60.0, 90.0, 120.0, 300.0])
    assert got == {"<25": 1, "25-50": 1, "50-75": 1, "75-100": 1, "100-150": 1, ">=250": 1}


@pytest.mark.asyncio
async def test_metrics_report_latency_histogram_and_queue(tmp_path):
    session, _ = _session(tmp_path)
    await session.start()
    empty = session.metrics
    assert empty["blocks"] == 0
    assert empty["latency_histogram"] == {}
    assert empty["p95_latency_ms"] == 0.0

    await session.push_pcm(np.zeros(24000, dtype=np.int16).tobytes())  # 5 blocks
    m = session.metrics
    assert m["blocks"] == 5
    assert sum(m["latency_histogram"].values()) == 5
    assert m["p95_latency_ms"] >= m["p50_latency_ms"] > 0.0
    assert m["max_latency_ms"] >= m["p95_latency_ms"]
    assert m["queue_depth"] == 0  # FakeEngine renders synchronously
    assert m["overflows"] == 0
    assert m["state"] == "speaking"
    await session.close()


# -- idle source-frame playback -----------------------------------------

class _IdleEngine(FakeEngine):
    """Serves solid-colour source frames ping-pong, like the real engine."""

    def __init__(self, n=3):
        super().__init__()
        self.n = n
        self.source = np.stack(
            [np.full((1280, 720, 3), 60 + i, np.uint8) for i in range(n)]
        )
        self.idle_calls: list[int] = []
        self.selections: list[int] = []

    class _Ctx:
        phase = 0
        emitted = 0

    def new_context(self, pack):
        return self._Ctx()

    async def idle_frame(self, pack, ctx, pos):
        from server.engine.musetalk_engine import cycle_index

        self.idle_calls.append(int(pos))
        return self.source[cycle_index(int(pos), self.n)]

    async def drive(self, pack, audio, sample_rate, *, context=None):
        frames = await super().drive(pack, audio, sample_rate, context=context)
        if context is not None:
            from server.engine.musetalk_engine import cycle_index

            first = int(context.emitted)
            # Mirror the real engine: base frames at phase + the cumulative
            # emitted anchor; emitted advances cumulatively and never resets.
            self.selections += [
                cycle_index(int(context.phase) + first + j, self.n)
                for j in range(len(frames))
            ]
            context.emitted = first + len(frames)
        return frames

    def idle_position(self, ctx):
        return int(getattr(ctx, "phase", 0)) + int(getattr(ctx, "emitted", 0))


class _ImageCapture(FrameBufferSink):
    keepalive = True

    def __init__(self):
        super().__init__()
        self.vals: list[int] = []

    async def push(self, image, pts):
        self.vals.append(int(image[0, 0, 0]))


@pytest.mark.asyncio
async def test_idle_track_plays_source_frames_in_cycle(tmp_path):
    engine = _IdleEngine(n=3)
    session, sink = _session(tmp_path, engine=engine, sink=_ImageCapture())
    await session.start()
    await asyncio.sleep(0.3)
    await session.close()
    # At least the first cycle must be source frames in ping-pong order.
    assert len(sink.vals) >= 5
    assert sink.vals[:5] == [60, 61, 62, 61, 60]


@pytest.mark.asyncio
async def test_speech_start_seeds_the_idle_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_LiveSink())
    await session.start()
    await asyncio.sleep(0.2)
    pos = session._idle_pos
    assert pos > 0
    await session.push_pcm(np.zeros(4800, np.int16).tobytes())
    assert session._ctx.phase == pos
    # A second, still-speaking block must not re-seed the phase.
    await session.push_pcm(np.zeros(4800, np.int16).tobytes())
    assert session._ctx.phase == pos
    await session.close()


@pytest.mark.asyncio
async def test_speech_end_reads_back_the_engine_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_LiveSink())
    await session.start()
    # 0.5 s: five 100 ms blocks, the fake engine emits 2 frames each -> emitted 10.
    await session.push_pcm(np.zeros(24000, np.int16).tobytes())
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.IDLE
    assert session._idle_pos == 10
    await session.close()


@pytest.mark.asyncio
async def test_idle_resumes_from_the_speech_end_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, sink = _session(tmp_path, engine=engine, sink=_ImageCapture())
    await session.start()
    await session.push_pcm(np.zeros(24000, np.int16).tobytes())  # 10 frames
    before = len(engine.idle_calls)
    session.maybe_idle(now=session.last_audio_at + 3.0)
    await asyncio.sleep(0.15)
    await session.close()
    # The idle track after speech asks for frames starting at position 10.
    assert engine.idle_calls[before] == 10


@pytest.mark.asyncio
async def test_static_fallback_does_not_advance_the_position(tmp_path):
    # Default FakeEngine has no idle frames (idle_frame -> None): the resting
    # canvas is pushed instead and the cycle position must stay put.
    session, _ = _session_with_canvas(tmp_path)
    await session.start()
    await asyncio.sleep(0.2)
    assert session.metrics["keepalive_frames"] > 0
    assert session._idle_pos == 0
    await session.close()


class _ClosedOnPush(_RecordingSink):
    keepalive = True

    async def push(self, image, pts):
        raise SinkClosed()


@pytest.mark.asyncio
async def test_sink_closed_does_not_advance_position(tmp_path):
    # The engine must supply a source frame so the loop reaches sink.push:
    # with the default FakeEngine there is no canvas in this pack, so the
    # push (and thus the SinkClosed path) would never be exercised.
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_ClosedOnPush())
    await session.start()
    await asyncio.sleep(0.15)
    assert session._idle_pos == 0
    assert session.metrics["dropped_frames"] > 0
    await session.close()


@pytest.mark.asyncio
async def test_second_speech_stretch_continues_without_double_counting(tmp_path):
    from server.engine.musetalk_engine import cycle_index

    engine = _IdleEngine(n=8)
    session, _ = _session(tmp_path, engine=engine, sink=_LiveSink())
    await session.start()
    # Turn 1: five 100 ms blocks -> 10 frames; phase seeded at 0, emitted 10.
    await session.push_pcm(np.zeros(24000, np.int16).tobytes())
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.IDLE
    assert session._idle_pos == 10
    # The idle track advanced three positions (10 -> 13) before speech resumes.
    session._idle_pos = 13
    await session.push_pcm(np.zeros(4800, np.int16).tobytes())  # 2 frames
    # Seamless edge: turn 2 starts on the frame the idle track would show
    # next, absolute position 13. The old absolute seed (phase = 13) composed
    # with the still-cumulative emitted (10) would select positions 23/24.
    assert session._ctx.phase == 3  # 13 - 10
    assert engine.selections[-2:] == [cycle_index(13, 8), cycle_index(14, 8)]
    # Readback after turn 2 stays on one continuous timeline: phase 3 + 12.
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session._idle_pos == 15
    await session.close()
