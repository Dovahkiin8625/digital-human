import numpy as np
from PIL import Image, ImageDraw

from server.engine.base import AvatarPack, DriveEngine, DriveFrame


class MockEngine(DriveEngine):
    name = "mock"

    def __init__(self, width: int = 720, height: int = 1280, fps: int = 25):
        self.width = width
        self.height = height
        self.fps = fps

    def describe(self) -> dict:
        return {"name": self.name, "gpu": False, "width": self.width, "height": self.height, "fps": self.fps}

    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate
        n_frames = max(1, round(duration * self.fps))
        samples_per_frame = sample_rate / self.fps
        base = Image.open(pack.source_image_path).convert("RGB").resize((self.width, self.height))
        scale = float(pack.config.get("mouth_scale", 1.0))
        frames: list[DriveFrame] = []
        for i in range(n_frames):
            start = int(i * samples_per_frame)
            end = min(len(audio), int((i + 1) * samples_per_frame))
            seg = audio[start:end] if end > start else np.zeros(1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            openness = min(1.0, rms * 4.0) * scale
            img = base.copy()
            draw = ImageDraw.Draw(img)
            cx, cy = self.width // 2, int(self.height * 0.72)
            mw, mh0 = int(self.width * 0.12), int(self.height * 0.015)
            mh = max(2, int(mh0 * (1 + openness * 6)))
            draw.ellipse((cx - mw, cy - mh, cx + mw, cy + mh), fill=(180, 40, 60))
            meter_h = int((self.height // 4) * openness)
            draw.rectangle((self.width - 40, self.height - meter_h - 20, self.width - 20, self.height - 20), fill=(40, 200, 90))
            frames.append(DriveFrame(pts=i / self.fps, image=np.asarray(img, dtype=np.uint8)))
        return frames
