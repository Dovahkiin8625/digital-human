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

    def save_source(self, avatar_id: str, stream, suffix: str) -> Path:
        """Persist the upload by streaming it, then reject anything unusable.

        The copy is byte-counted: a body larger than
        ``avatar_max_upload_bytes`` is aborted mid-copy with a 413 and the
        partial file is unlinked, so an oversized upload is never pulled
        fully into RAM or written fully to disk before it is refused.
        Image/video validation then runs on the saved file; any failure
        unlinks the file and removes the now-empty avatar directory -- a
        rejected upload gets no Avatar row, so nothing else ever owns it.
        """
        settings = get_settings()
        d = self.avatar_dir(avatar_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"source{suffix or '.png'}"
        try:
            written = 0
            with path.open("wb") as out:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > settings.avatar_max_upload_bytes:
                        raise AppError(
                            "file_too_large",
                            f"upload is larger than {settings.avatar_max_upload_bytes} bytes",
                            413,
                        )
                    out.write(chunk)
            if video.is_video_suffix(suffix):
                self._validate_video(path)
            else:
                self._validate_image(path)
        except Exception:
            # Broadened from AppError: a non-typed escape (e.g. a raw
            # cv2.error from a validator) must not leak the file either.
            path.unlink(missing_ok=True)
            try:
                d.rmdir()  # empty at upload time; non-empty if another write raced
            except OSError:
                pass
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
