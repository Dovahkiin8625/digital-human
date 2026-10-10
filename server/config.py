from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    drive_engine: str = "musetalk"
    avatar_pipeline: str = "musetalk"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/digital-human.db"
    admin_password: str = "admin123"
    admin_token: str = "dev-admin-token"
    session_idle_timeout: float = 60.0
    session_sweep_interval: float = 10.0
    frame_width: int = 720
    frame_height: int = 1280
    fps: int = 25
    sample_rate: int = 48000

    # Video-driven avatars. A driver video is decoded at the output fps and
    # capped so its frames can stay resident: 15 s at 25 fps is 375 frames.
    avatar_max_video_seconds: float = 15.0
    avatar_max_video_frames: int = 375
    # Video avatars keep ~1 GB of decoded frames resident each, so the engine's
    # per-avatar LRU is smaller than it was when every avatar was one frame.
    avatar_cache_size: int = 2
    # Uploads are streamed to disk and counted as they arrive; a body past
    # this many bytes is refused mid-copy with 413. Generous above a 15 s
    # clip even at a phone high bitrate (100 Mbps * 15 s ~= 188 MB).
    avatar_max_upload_bytes: int = 300_000_000

    # MiniMax text-to-speech, used by the console's text-driven debug page.
    # Empty api_key means "not configured": the endpoint returns 503 rather than
    # calling out. group_id is optional -- newer accounts authenticate on the
    # bearer token alone, and MiniMax ignores the query param when it is absent.
    minimax_api_key: str = ""
    minimax_group_id: str = ""
    minimax_base_url: str = "https://api.minimaxi.com"
    minimax_model: str = "speech-02-hd"
    minimax_voice_id: str = "male-qn-qingse"
    # MiniMax tops out at 44100 Hz, so this is never the render session's 48000;
    # the browser resamples onto the WS audio channel (webrtc.ts).
    minimax_sample_rate: int = 32000

    model_config = SettingsConfigDict(env_prefix="DH_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
