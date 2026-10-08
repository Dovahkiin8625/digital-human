import time
from enum import Enum

import numpy as np

from server.core.audio_ingest import AudioIngest
from server.core.errors import AppError
from server.core.stream_sink import SinkClosed, StreamSink
from server.engine.base import AvatarPack, DriveEngine

IDLE_AFTER = 2.0


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
        self._pts = 0.0
        self._metrics = {"frames_out": 0, "dropped_frames": 0, "fps": 0.0, "avg_latency_ms": 0.0}

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    async def start(self) -> None:
        await self.engine.prepare(self.pack)
        self.state = SessionState.LISTENING

    def maybe_idle(self, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.LISTENING

    async def push_pcm(self, data: bytes) -> None:
        if self.state in (SessionState.CLOSED, SessionState.ERROR):
            raise AppError("session_closed", f"session {self.session_id} is {self.state.value}", 409)
        self.last_audio_at = time.monotonic()
        self.state = SessionState.SPEAKING
        for block in self._ingest.push_pcm(data):
            await self._drive_block(block)

    async def _drive_block(self, block: np.ndarray) -> None:
        t0 = time.perf_counter()
        frames = await self.engine.drive(self.pack, block, self.sample_rate)
        for frame in frames:
            frame.pts = self._pts
            self._pts += 1.0 / self.fps
            try:
                await self.sink.push(frame.image, frame.pts)
                self._metrics["frames_out"] += 1
            except SinkClosed:
                self._metrics["dropped_frames"] += 1
                return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prev = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = latency_ms if prev == 0 else prev * 0.8 + latency_ms * 0.2
        self._metrics["fps"] = float(self.fps)

    def mark_error(self, message: str) -> None:
        self.state = SessionState.ERROR
        self.error = message[:1024]

    async def close(self) -> None:
        await self.engine.close()
        await self.sink.close()
        self.state = SessionState.CLOSED
