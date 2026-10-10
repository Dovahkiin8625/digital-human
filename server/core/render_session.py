import asyncio
import contextlib
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path

import numpy as np

from server.core.audio_ingest import AudioIngest
from server.core.errors import AppError
from server.core.stream_sink import SinkClosed, StreamSink
from server.engine.base import AvatarPack, DriveEngine

IDLE_AFTER = 2.0

#: Blocks retained for the latency summary: 60 s worth at 100 ms blocks.
LATENCY_WINDOW = 600
#: Histogram edges in ms. A block is produced from 100 ms of audio, so 100 ms
#: is the real-time boundary; the buckets cluster there.
LATENCY_EDGES = (25.0, 50.0, 75.0, 100.0, 150.0, 250.0)


def _histogram(samples) -> dict[str, int]:
    labels = (
        [f"<{LATENCY_EDGES[0]:.0f}"]
        + [f"{a:.0f}-{b:.0f}" for a, b in zip(LATENCY_EDGES, LATENCY_EDGES[1:])]
        + [f">={LATENCY_EDGES[-1]:.0f}"]
    )
    counts = [0] * len(labels)
    for x in samples:
        for i, edge in enumerate(LATENCY_EDGES):
            if x < edge:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return {label: n for label, n in zip(labels, counts) if n}


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    CLOSED = "closed"
    ERROR = "error"


class RenderSession:
    def __init__(
        self,
        session_id: str,
        pack: AvatarPack,
        engine: DriveEngine,
        sink: StreamSink,
        sample_rate: int = 48000,
        fps: int = 25,
    ):
        self.session_id = session_id
        self.pack = pack
        self.engine = engine
        self.sink = sink
        self.sample_rate = sample_rate
        self.fps = fps
        self.state = SessionState.IDLE
        self.error: str | None = None
        self.last_audio_at = time.monotonic()
        self._ingest = AudioIngest(sample_rate=sample_rate, block_ms=100)
        self._audio_pts = 0.0
        # Per-session engine state (rolling audio window, frame accumulator).
        # Built in start(), once prepare() has made the avatar's artifacts
        # available; every drive() call for this session gets the same object.
        self._ctx: object | None = None
        self._metrics = {
            "frames_out": 0,
            "dropped_frames": 0,
            "keepalive_frames": 0,
            "fps": 0.0,
            "avg_latency_ms": 0.0,
        }
        self._blocks = 0
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        # `metrics` is read from a *sync* FastAPI endpoint (admin_list_sessions),
        # which FastAPI runs in a threadpool, while the event loop appends to
        # `_latencies` in _drive_block. Iterating a deque that another thread
        # mutates raises "deque mutated during iteration" and 500s the admin
        # sessions poll, so both sides take this lock. A threading.Lock, not an
        # asyncio one: the reading side is synchronous and cannot await.
        self._latencies_lock = threading.Lock()
        self._keepalive_task: asyncio.Task | None = None
        self._idle_frame: np.ndarray | None = None
        self._idle_frame_read = False
        # Ping-pong position of the next idle frame. Advanced per successful
        # source-frame tick; seeded INTO the engine context when speech starts
        # and re-read FROM it when speech ends, so idle and speech share one
        # continuous position -- no jump at either boundary.
        self._idle_pos = 0
        # Serializes the entry points that drive frames and advance pts/metrics.
        # push_pcm (audio WS / RTP track) and close (DELETE / idle sweep) can run
        # concurrently; a real GPU engine's awaited drive() would otherwise let
        # their frame sequences interleave.
        self._lock = asyncio.Lock()

    @property
    def metrics(self) -> dict:
        out = dict(self._metrics)
        out["blocks"] = self._blocks
        out["state"] = self.state.value
        out["queue_depth"] = self.engine.pending()
        out["overflows"] = self.sink.overflows
        # Snapshot under the lock so the percentile and histogram passes below
        # never iterate the deque while _drive_block appends to it.
        with self._latencies_lock:
            latencies = list(self._latencies)
        if latencies:
            arr = np.asarray(latencies, dtype=np.float64)
            out["p50_latency_ms"] = float(np.percentile(arr, 50))
            out["p95_latency_ms"] = float(np.percentile(arr, 95))
            out["max_latency_ms"] = float(arr.max())
        else:
            out["p50_latency_ms"] = out["p95_latency_ms"] = out["max_latency_ms"] = 0.0
        out["latency_histogram"] = _histogram(latencies)
        return out

    async def start(self) -> None:
        await self.engine.prepare(self.pack)
        self._ctx = self.engine.new_context(self.pack)
        self.state = SessionState.LISTENING
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def maybe_idle(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.IDLE
            self._idle_pos = int(self.engine.idle_position(self._ctx))

    # -- idle keep-alive -------------------------------------------------
    def _static_frame(self) -> np.ndarray | None:
        """The avatar's resting canvas, read once. None disables keep-alive.

        Every pipeline writes this file, and it is the same frame MuseTalk
        composites onto, so an idle session shows the avatar rather than a
        frozen or blank track. A pack without it (the mock engine's, in tests)
        simply never sends keep-alive frames.
        """
        if not self._idle_frame_read:
            self._idle_frame_read = True
            try:
                from PIL import Image

                path = Path(self.pack.root) / self.pack.prep.get("base_image", "normalized.png")
                self._idle_frame = np.array(Image.open(path).convert("RGB"))
            except Exception:  # noqa: BLE001 -- a missing canvas must not break the session
                self._idle_frame = None
        return self._idle_frame

    async def _keepalive_loop(self) -> None:
        """Keep a live video track fed while the session has nothing to say.

        An idle session drives no frames, so the track's queue drains and the
        sender blocks on recv() -- the receiver sees a frozen picture and the
        RTP stream stops. Pushing the resting canvas at the nominal rate keeps
        the stream's timing honest until speech resumes, and advances the
        session clock by the same amount, so there is no pts jump when it does.
        """
        period = 1.0 / self.fps
        started = time.monotonic()
        tick = 0
        while True:
            tick += 1
            await asyncio.sleep(max(0.0, started + tick * period - time.monotonic()))
            self.maybe_idle()
            # Check CLOSED/ERROR out here, not only under the lock below. The
            # lock is entered only when this sink wants keep-alive *and* a
            # resting frame exists, so a session on a NullSink, or one whose
            # pack has no normalized.png, would otherwise skip that check
            # forever -- a 25 Hz no-op task surviving close()/stop()/shutdown.
            if self.state in (SessionState.CLOSED, SessionState.ERROR):
                return
            if self.state is SessionState.SPEAKING or not self.sink.keepalive:
                continue
            source_frame = await self.engine.idle_frame(
                self.pack, self._ctx, int(self._idle_pos)
            )
            # Static fallback only when the engine supplies no source frame;
            # both are fetched before the lock, below.
            frame = None if source_frame is not None else self._static_frame()
            if source_frame is None and frame is None:
                continue
            async with self._lock:
                if self.state in (SessionState.CLOSED, SessionState.ERROR):
                    return
                # Re-check under the lock: speech may have started between the
                # pre-check above and here, and a frame pushed between two
                # speech frames is a visible flicker.
                if self.state is SessionState.SPEAKING:
                    continue
                try:
                    pushed = source_frame if source_frame is not None else frame
                    await self.sink.push(pushed, self._audio_pts)
                except SinkClosed:
                    self._metrics["dropped_frames"] += 1
                    continue
                # Counted apart from frames_out: nothing drove this frame, and
                # folding it in would make the driving pipeline's fps and
                # latency look better the longer a session sat unused.
                self._metrics["keepalive_frames"] += 1
                self._audio_pts += period
                if source_frame is not None:
                    # Only source frames occupy cycle positions; a static
                    # fallback must not move the phase the next speech seeds.
                    self._idle_pos += 1

    async def push_pcm(self, data: bytes) -> None:
        async with self._lock:
            if self.state in (SessionState.CLOSED, SessionState.ERROR):
                raise AppError("session_closed", f"session {self.session_id} is {self.state.value}", 409)
            if self.state is not SessionState.SPEAKING:
                # First audio after idle: hand the idle position to the engine
                # so its first speech frame is the frame the idle track would
                # have shown next. Duck-typed: a mock engine's context is None.
                if self._ctx is not None and hasattr(self._ctx, "phase"):
                    self._ctx.phase = int(self._idle_pos)
            self.last_audio_at = time.monotonic()
            self.state = SessionState.SPEAKING
            for block in self._ingest.push_pcm(data):
                await self._drive_block(block)

    async def flush(self) -> None:
        """Drain the ingest's partial tail block, driving any frames it yields."""
        async with self._lock:
            for block in self._ingest.flush():
                await self._drive_block(block)

    async def _drive_block(self, block: np.ndarray) -> None:
        t0 = time.perf_counter()
        block_duration = len(block) / self.sample_rate
        block_start = self._audio_pts
        self._audio_pts += block_duration
        try:
            frames = await self.engine.drive(
                self.pack, block, self.sample_rate, context=self._ctx
            )
        except Exception as exc:
            self.mark_error(str(exc))
            raise
        for j, frame in enumerate(frames):
            frame.pts = block_start + j * block_duration / len(frames)
            try:
                await self.sink.push(frame.image, frame.pts)
                self._metrics["frames_out"] += 1
            except SinkClosed:
                self._metrics["dropped_frames"] += 1
                return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._blocks += 1
        with self._latencies_lock:
            self._latencies.append(latency_ms)
        prev = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = latency_ms if prev == 0 else prev * 0.8 + latency_ms * 0.2
        self._metrics["fps"] = float(self.fps)

    def mark_error(self, message: str) -> None:
        self.state = SessionState.ERROR
        self.error = message[:1024]

    def stop(self) -> None:
        """Cancel the keep-alive task without awaiting it, for teardown paths."""
        task, self._keepalive_task = self._keepalive_task, None
        if task is not None:
            task.cancel()

    async def close(self) -> None:
        # Cancel before taking the lock: the loop may be holding it while it
        # awaits a sink push, and it must not be left running past CLOSED.
        task, self._keepalive_task = self._keepalive_task, None
        if task is not None:
            task.cancel()
            # A session can be closed from a loop other than the one that
            # started it -- the admin's sweep-now endpoint runs sweep_idle under
            # asyncio.run -- and a task may only be awaited on its own loop.
            # Cancelling is enough for the foreign case: that loop reaps it.
            if task.get_loop() is asyncio.get_running_loop():
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        async with self._lock:
            for block in self._ingest.flush():
                await self._drive_block(block)
            # The DriveEngine is shared across sessions and owned by SessionManager
            # (closed once via SessionManager.shutdown); a session only closes its sink.
            await self.sink.close()
            self.state = SessionState.CLOSED
