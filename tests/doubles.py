"""Test doubles for the digital-human model.

The server ships exactly one model (MuseTalk). These doubles stand in for
it so the whole test suite can run in the torch-free ``.venv``: no GPU, no
weights, no torch.

They live here rather than in ``server/`` on purpose. A second
implementation in the production package would be a second deployment mode
nobody asked for -- and the server would have to carry a switch to choose
between them, which is exactly what the retired ``DH_DRIVE_ENGINE`` /
``DH_AVATAR_PIPELINE`` env vars were.

The behaviour of both doubles is frozen: the suite asserts on their exact
frame arithmetic (``round(duration * fps)`` per 100 ms block), on the pack
they write (``{"crop": {"mode": "center"}}`` plus a 720x1280
``normalized.png``), and on the stage names they report. Changing either
number breaks tests that were written against them, not against MuseTalk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from server.avatar.pipeline import AvatarPipeline, PipelineError
from server.config import get_settings
from server.engine.base import AvatarPack, DriveEngine, DriveFrame


class FakeEngine(DriveEngine):
    """A CPU-only drive engine that paints a mouth opening on the source photo.

    Stateless: it ignores the per-session context entirely, and it renders
    synchronously, so a test can assert on the frame count right after the
    call returns.
    """

    name = "fake"

    def __init__(self, width: int = 720, height: int = 1280, fps: int = 25):
        self.width = width
        self.height = height
        self.fps = fps

    def describe(self) -> dict:
        # The Models admin tab reads ``models`` to decide which buttons to
        # show. This double has no weights, so the entry is permanently
        # "loaded, can't unload" -- the operator still sees a row confirming
        # the current engine is the CPU placeholder.
        return {
            "name": self.name,
            "gpu": False,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "models": [
                {
                    "id": "fake_drive",
                    # Unified ModelInfo shape: ``kind`` lets the Models tab
                    # render engine rows alongside portrait-animator rows
                    # in one table (the row's `type` column still uses the
                    # historical "drive_runtime" string for backward compat).
                    "kind": "drive_runtime",
                    "label_zh": "Fake 驱动运行时",
                    "type": "drive_runtime",
                    "components": [],
                    "source": None,
                    "estimated_bytes": 0,
                    "loaded": True,
                    # No weights to mount; the load button stays disabled.
                    # ``can_load`` mirrors the engine's load capability so the
                    # reload-after-unload bug fix is symmetric between the
                    # double and the real engine.
                    "can_load": False,
                    "can_unload": False,
                    "loaded_at": None,
                }
            ],
        }

    async def drive(
        self,
        pack: AvatarPack,
        audio: np.ndarray,
        sample_rate: int,
        *,
        context: object | None = None,
    ) -> list[DriveFrame]:
        # Stateless: the double ignores per-session context entirely.
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate
        n_frames = max(1, round(duration * self.fps))
        samples_per_frame = sample_rate / self.fps
        base = Image.open(pack.source_image_path).convert("RGB").resize((self.width, self.height))
        frames: list[DriveFrame] = []
        for i in range(n_frames):
            start = int(i * samples_per_frame)
            end = min(len(audio), int((i + 1) * samples_per_frame))
            seg = audio[start:end] if end > start else np.zeros(1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
            openness = min(1.0, rms * 4.0)
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


class FakePipeline(AvatarPipeline):
    """Writes a placeholder pack: a centre-cropped 720x1280 canvas.

    The stages it reports are the same four the real pipeline reports, so
    a job's timeline looks the same either way.
    """

    MIN_SIZE = 256
    def process(
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress,
    ) -> dict:
        from server.avatar import video

        progress("detect_face", 10)
        source_path = Path(source_path)
        if video.is_video_suffix(source_path.suffix):
            settings = get_settings()
            frames, _ = video.decode_frames(
                source_path,
                target_fps=settings.fps,
                max_frames=1,
                canvas=(settings.frame_width, settings.frame_height),
            )
            if frames.shape[0] == 0:
                raise PipelineError("video has no decodable frames")
            img = Image.fromarray(frames[0])
        else:
            img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}")
        progress("align", 40)
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        cropped = img.crop((left, top, left + side, top + side))
        progress("normalize", 70)
        out_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (720, 1280), (16, 16, 24))
        resized = cropped.resize((720, 720))
        canvas.paste(resized, (0, 280))
        canvas.save(out_dir / "normalized.png")
        progress("build_pack", 100)
        return {"crop": {"mode": "center"}}


def install_doubles(monkeypatch) -> None:
    """Point the server's model factories at the doubles.

    ``server.main`` imports ``create_engine`` inside ``create_app`` and
    ``create_pipeline`` is looked up as a module attribute at call time, so
    patching the two module attributes swaps the whole model -- engine and
    pipeline together, which is the invariant the registry exists to keep.
    """
    monkeypatch.setattr("server.engine.base.create_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr("server.avatar.pipeline.create_pipeline", lambda *a, **k: FakePipeline())
