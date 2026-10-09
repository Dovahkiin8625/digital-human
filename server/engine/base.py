from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.config import Settings, get_settings


class EngineUnavailable(Exception):
    pass


def gpu_available() -> bool:
    """Whether a CUDA device is usable. Never raises, never loads a weight.

    torch only exists in the GPU venv, so this must tolerate ImportError: the
    torch-free venv is a supported configuration (mock engine + test suite).
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class AvatarPack:
    avatar_id: str
    root: Path
    source_image_path: Path
    config: dict = field(default_factory=dict)
    # Small scalars/paths from the pipeline's `pack.json` "musetalk" block
    # (bbox, crop size, mask_crop_box, sidecar dir). Never holds tensors --
    # the engine loads those lazily from `root`.
    prep: dict = field(default_factory=dict)


@dataclass
class DriveFrame:
    pts: float
    image: np.ndarray


class DriveEngine(ABC):
    name: str = "base"

    async def prepare(self, pack: AvatarPack) -> None:
        pass

    def new_context(self, pack: AvatarPack) -> object | None:
        """Per-session streaming state, or None for a stateless engine.

        Called once per session, after prepare(); whatever it returns is handed
        back to every drive() call for that session. A streaming engine keeps
        its rolling audio window and its frame accumulator here rather than in
        a dict keyed by session id, so nothing can leak when a session is
        dropped without an explicit teardown.
        """
        return None

    @abstractmethod
    async def drive(
        self,
        pack: AvatarPack,
        audio: np.ndarray,
        sample_rate: int,
        *,
        context: object | None = None,
    ) -> list[DriveFrame]:
        ...

    async def warmup(self) -> None:
        """Load weights / compile kernels ahead of the first session.

        Default no-op. The lifespan runs this as a background task and only
        logs a failure, so a box with no GPU still boots.
        """
        return None

    async def preview(self, pack: AvatarPack, level: float = 0.5) -> np.ndarray:
        """A single frame for the admin debug page's openness slider."""
        await self.prepare(pack)
        audio = _tone(level)
        frames = await self.drive(
            pack, audio, get_settings().sample_rate, context=self.new_context(pack)
        )
        if not frames:
            # A 200 ms tone always yields frames today, but the admin preview
            # endpoint must not 500 on an engine that returns none.
            raise EngineUnavailable(
                f"engine {self.name} produced no frames for the preview"
            )
        return frames[0].image

    def pending(self) -> int:
        """Jobs queued but not yet rendered. 0 for a synchronous engine.

        A streaming engine renders on its own worker thread, so a session that
        is outrunning it shows up here before it shows up as latency.
        """
        return 0

    def describe(self) -> dict:
        return {"name": self.name, "gpu": False}

    async def close(self) -> None:
        pass


def _tone(level: float, ms: int = 200, hz: float = 160.0) -> np.ndarray:
    """A short voiced-range sine, for previewing a mouth shape at `level`."""
    settings = get_settings()
    n = settings.sample_rate * ms // 1000
    t = np.arange(n, dtype=np.float32) / settings.sample_rate
    return (np.sin(2 * np.pi * hz * t) * float(level)).astype(np.float32)


def create_engine(name: str | None = None, settings: Settings | None = None) -> DriveEngine:
    settings = settings or get_settings()
    name = name or settings.drive_engine
    if name == "mock":
        from server.engine.mock_engine import MockEngine

        return MockEngine(settings.frame_width, settings.frame_height, settings.fps)
    if name == "musetalk":
        from server.engine.musetalk_engine import MuseTalkEngine

        return MuseTalkEngine()
    raise EngineUnavailable(f"unknown engine: {name}")
