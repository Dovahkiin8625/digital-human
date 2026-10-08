import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from PIL import Image


class PipelineError(Exception):
    pass


class AvatarPipeline(ABC):
    @abstractmethod
    def process(self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict:
        ...


class MockAvatarPipeline(AvatarPipeline):
    MIN_SIZE = 256

    def process(self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict:
        progress("detect_face", 10)
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
        return {"mouth_scale": 1.0, "expression_intensity": 1.0, "crop": {"mode": "center"}}


def create_pipeline(name: str) -> AvatarPipeline:
    if name == "mock":
        return MockAvatarPipeline()
    raise PipelineError(f"unknown pipeline: {name}")


def run_avatar_job(db, store, pipeline: AvatarPipeline, job_id: str) -> None:
    from server.db.models import Avatar, AvatarJob

    job = db.get(AvatarJob, job_id)
    avatar = db.get(Avatar, job.avatar_id)

    def progress(stage: str, pct: int) -> None:
        job.stage = stage
        job.progress = pct
        db.commit()

    try:
        avatar.status = "processing"
        db.commit()
        config = pipeline.process(
            avatar.id, Path(avatar.source_image_path), store.avatar_dir(avatar.id), progress
        )
        pack_json = store.avatar_dir(avatar.id) / "pack.json"
        pack_json.write_text(json.dumps(config))
        avatar.pack_path = str(pack_json)
        avatar.config_json = config
        avatar.status = "ready"
        db.commit()
    except Exception as exc:  # noqa: BLE001 — 失败必须落库可见
        db.rollback()
        avatar.status = "failed"
        job.error = str(exc)[:1024]
        db.commit()
