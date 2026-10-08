from abc import ABC, abstractmethod

import numpy as np


class SinkClosed(Exception):
    pass


class StreamSink(ABC):
    @abstractmethod
    async def push(self, image: np.ndarray, pts: float) -> None:
        ...

    async def close(self) -> None:
        pass


class NullSink(StreamSink):
    async def push(self, image: np.ndarray, pts: float) -> None:
        pass


class FrameBufferSink(StreamSink):
    def __init__(self):
        self.latest: tuple[float, np.ndarray] | None = None
        self.dropped = 0

    async def push(self, image: np.ndarray, pts: float) -> None:
        if self.latest is not None:
            self.dropped += 1
        self.latest = (pts, image)

    async def close(self) -> None:
        self.latest = None
