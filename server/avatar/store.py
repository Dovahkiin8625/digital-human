import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from server.avatar.pipeline import PipelineError
from server.db.models import Avatar
from server.engine.base import AvatarPack


class AvatarStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.data_dir / "avatars" / avatar_id

    def save_source_image(self, avatar_id: str, data: bytes, suffix: str) -> Path:
        d = self.avatar_dir(avatar_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"source{suffix or '.png'}"
        path.write_bytes(data)
        try:
            Image.open(path).verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise PipelineError("source image is not a decodable image") from exc
        return path

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
        )

    def delete_assets(self, avatar_id: str) -> None:
        d = self.avatar_dir(avatar_id)
        if d.exists():
            shutil.rmtree(d)
