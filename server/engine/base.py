from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class EngineUnavailable(Exception):
    pass


def gpu_available() -> bool:
    """Whether a CUDA device is usable. Never raises, never loads a weight.

    torch only exists in the GPU venv, so this must tolerate ImportError: the
    torch-free venv is a supported configuration for the test suite.
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

    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The source frame at ping-pong position `pos` for an idle track.

        Returns an HWC RGB array, or None when this engine/pack cannot supply
        source frames; the caller then falls back to its resting canvas. A
        stateless engine returns None, which keeps the pre-existing static
        keep-alive behaviour.
        """
        return None

    def idle_position(self, context: object | None) -> int:
        """Ping-pong cycle position after the most recently used frame."""
        return 0

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


def create_engine(name: str | None = None) -> DriveEngine:
    """Build the drive engine of a registered digital-human model.

    ``name`` defaults to the fallback model (see
    ``server.model_registry``). The engine and the avatar pipeline always
    come from the same model, so a pack can never be written by one model
    and driven by another.

    Raises :class:`EngineUnavailable` for a name that is not registered,
    and for a model the *client* renders -- it has no server engine.
    """
    from server.model_registry import ModelUnavailable, get_model

    try:
        model = get_model(name)
    except ModelUnavailable as exc:
        raise EngineUnavailable(str(exc)) from exc
    if model.engine_factory is None:
        raise EngineUnavailable(
            f"model {model.name!r} is rendered by the client; "
            "it has no server drive engine"
        )
    return model.engine_factory()
