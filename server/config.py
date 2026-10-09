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

    model_config = SettingsConfigDict(env_prefix="DH_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
