from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.config import Settings, get_settings


class EngineUnavailable(Exception):
    pass


@dataclass
class AvatarPack:
    avatar_id: str
    root: Path
    source_image_path: Path
    config: dict = field(default_factory=dict)


@dataclass
class DriveFrame:
    pts: float
    image: np.ndarray


class DriveEngine(ABC):
    name: str = "base"

    async def prepare(self, pack: AvatarPack) -> None:
        pass

    @abstractmethod
    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        ...

    def describe(self) -> dict:
        return {"name": self.name, "gpu": False}

    async def close(self) -> None:
        pass


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
