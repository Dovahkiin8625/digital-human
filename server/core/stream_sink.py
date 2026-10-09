from abc import ABC, abstractmethod

import numpy as np


class SinkClosed(Exception):
    pass


class StreamSink(ABC):
    #: Whether an idle session should keep pushing static frames. True only for
    #: sinks backed by a live video track, where a silent producer freezes the
    #: receiver; a NullSink has no transport to keep alive.
    keepalive = False
    #: Frames this sink had to drop because it could not keep up.
    overflows = 0

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
