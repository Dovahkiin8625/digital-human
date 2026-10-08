import numpy as np


class AudioIngest:
    def __init__(self, sample_rate: int = 48000, block_ms: int = 100):
        self.sample_rate = sample_rate
        self.block_samples = sample_rate * block_ms // 1000
        self._buf = bytearray()

    def _drain_blocks(self) -> list[np.ndarray]:
        blocks = []
        while len(self._buf) >= self.block_samples * 2:
            raw = bytes(self._buf[: self.block_samples * 2])
            del self._buf[: self.block_samples * 2]
            blocks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        return blocks

    def push_pcm(self, data: bytes) -> list[np.ndarray]:
        self._buf.extend(data)
        return self._drain_blocks()

    def flush(self) -> list[np.ndarray]:
        blocks = self._drain_blocks()
        if len(self._buf) >= 2:
            raw = bytes(self._buf)
            self._buf.clear()
            blocks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        return blocks
