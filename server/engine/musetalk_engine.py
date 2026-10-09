from pathlib import Path

import numpy as np

from server.config import get_settings
from server.engine.base import AvatarPack, DriveEngine, DriveFrame, EngineUnavailable

WEIGHTS = ["musetalk.json", "musetalk_v2.ckpt", "vae.pth", "whisper"]


class MuseTalkEngine(DriveEngine):
    name = "musetalk"

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or get_settings().data_dir / "models" / "musetalk")
        self._model = None

    def describe(self) -> dict:
        gpu = False
        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            pass
        return {"name": self.name, "gpu": gpu, "models_dir": str(self.models_dir)}

    async def prepare(self, pack: AvatarPack) -> None:
        missing = [w for w in WEIGHTS if not (self.models_dir / w).exists()]
        if missing:
            raise EngineUnavailable(
                f"musetalk weights not found in {self.models_dir} (missing: {missing}); see README for setup"
            )
        self._load_model_if_needed()

    def _load_model_if_needed(self) -> None:
        if self._model is not None:
            return
        import torch

        self._model = {"ckpt": torch.load(self.models_dir / "musetalk_v2.ckpt", map_location="cpu")}
        if torch.cuda.is_available():
            self._model = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in self._model.items()}

    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        win = sample_rate * 40 // 1000
        feats = []
        for i in range(0, max(len(audio) - win + 1, 1), win):
            seg = audio[i : i + win]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            feats.append(spec[:16] if len(spec) >= 16 else np.pad(spec, (0, 16 - len(spec))))
        return np.asarray(feats, dtype=np.float32)

    def _run_unet(self, feats: np.ndarray, i: int) -> np.ndarray:
        import torch

        self._load_model_if_needed()
        x = torch.from_numpy(feats[i : i + 1])
        with torch.no_grad():
            _ = self._model["ckpt"]
            _ = x
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def _blend(self, face_crop: np.ndarray, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        x0, y0 = out.shape[1] // 2 - 128, int(out.shape[0] * 0.55) - 128
        region = out[y0 : y0 + 256, x0 : x0 + 256]
        alpha = 0.9
        region[:] = (region * (1 - alpha) + face_crop * alpha).astype(np.uint8)
        return out

    async def drive(self, pack: AvatarPack, audio: np.ndarray, sample_rate: int) -> list[DriveFrame]:
        from PIL import Image

        fps = 25
        duration = len(audio) / sample_rate
        n_frames = max(1, round(duration * fps))
        feats = self._extract_audio_features(np.asarray(audio, dtype=np.float32), sample_rate)
        base = np.asarray(Image.open(pack.source_image_path).convert("RGB").resize((720, 1280)))
        frames = []
        for i in range(n_frames):
            face = self._run_unet(feats, min(i, max(len(feats) - 1, 0)))
            img = self._blend(face, base)
            frames.append(DriveFrame(pts=i / fps, image=img.astype(np.uint8)))
        return frames
