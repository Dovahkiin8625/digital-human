import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from server.avatar import video
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar
from server.engine.base import AvatarPack


class AvatarStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.data_dir / "avatars" / avatar_id

    def save_source(self, avatar_id: str, data: bytes, suffix: str) -> Path:
        """Persist the upload and reject anything unusable, before any job runs.

        An image is validated by decoding it; a video by probing it, which also
        enforces the duration/frame cap here rather than inside the background
        job -- a video that is too long should 422 at upload, not become a
        `failed` avatar ten seconds later.
        """
        d = self.avatar_dir(avatar_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"source{suffix or '.png'}"
        path.write_bytes(data)
        try:
            if video.is_video_suffix(suffix):
                self._validate_video(path)
            else:
                self._validate_image(path)
        except AppError:
            # Nothing will own this file: a rejected upload never gets an Avatar
            # row, so no job and no delete_assets() will ever come back for it.
            # Previously an invalid image leaked a few KB; a too-long video is
            # tens of MB.
            path.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _validate_image(path: Path) -> None:
        try:
            Image.open(path).verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise AppError("invalid_image", "source image is not a decodable image", 422) from exc

    def _validate_video(self, path: Path) -> None:
        settings = get_settings()
        try:
            info = video.probe_video(path)
        except video.VideoError as exc:
            raise AppError("invalid_video", str(exc), 422) from exc
        out_frames = int(info["duration"] * settings.fps)
        if info["duration"] > settings.avatar_max_video_seconds or out_frames > settings.avatar_max_video_frames:
            raise AppError(
                "video_too_long",
                f"video is {info['duration']:.1f}s (~{out_frames} frames); "
                f"limit is {settings.avatar_max_video_seconds:.0f}s / {settings.avatar_max_video_frames} frames",
                422,
            )

    def load_pack(self, avatar: Avatar) -> AvatarPack:
        root = self.avatar_dir(avatar.id)
        pack_json = root / "pack.json"
        config = json.loads(pack_json.read_text()) if pack_json.exists() else {}
        source = Path(avatar.source_image_path)
        normalized = root / "normalized.png"
        return AvatarPack(
            avatar_id=avatar.id,
            root=root,
            source_image_path=normalized if normalized.exists() else source,
            config=config,
            prep=config.get("musetalk", {}),
        )

    def delete_assets(self, avatar_id: str) -> None:
        d = self.avatar_dir(avatar_id)
        if d.exists():
            shutil.rmtree(d)
