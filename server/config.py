from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    drive_engine: str = "musetalk"
    avatar_pipeline: str = "musetalk"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/digital-human.db"
    session_idle_timeout: float = 60.0
    session_sweep_interval: float = 10.0
    frame_width: int = 720
    frame_height: int = 1280
    fps: int = 25
    sample_rate: int = 48000
    # Outbound H.264. aiortc's OpenH264 defaults to 1 Mbps with a 3 Mbps REMB
    # ceiling; at 720x1280@25 the talking mouth is the busiest region in the
    # frame and 1 Mbps turns it into blocky noise. Raised to 3.5/4 Mbps at
    # startup (see server.core.webrtc.configure_h264_bitrate).
    video_bitrate: int = 3_500_000
    video_max_bitrate: int = 4_000_000

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
    # voice_id is *not* here: the operator picks it per session in the debug
    # page, so the synthesise endpoint takes it as a required argument rather
    # than reading a server-wide default.
    minimax_api_key: str = ""
    minimax_group_id: str = ""
    minimax_base_url: str = "https://api.minimaxi.com"
    minimax_model: str = "speech-02-hd"
    # MiniMax tops out at 44100 Hz, so this is never the render session's 48000;
    # the browser resamples onto the WS audio channel (webrtc.ts).
    minimax_sample_rate: int = 32000

    # NOTE: LLM config (api_key / base_url / model) lives entirely in the
    # SQLite ``settings`` table and is configured from the Models tab.
    # There are deliberately no ``DH_LLM_*`` env defaults here -- env vars
    # are a deployment concern, but the LLM is a per-box operator choice
    # that's expected to be wired through the UI. Removing the env shim
    # also keeps a leaked ``DH_LLM_API_KEY`` from silently turning the chat
    # endpoint on in environments where the operator hasn't opted in.

    # Static credential for the business-plane endpoints a *remote* client
    # reaches -- today that is the Electron pet, which renders locally and
    # calls back for models and the Cubism runtime. Empty disables the
    # check, which is what a single-operator box behind the admin console
    # wants. The admin plane itself has no auth since the no-tenant refactor
    # and is a separate problem.
    v1_api_token: str = ""

    # Live2D. `live2d_sdk_dir` is read by the server only to serve the SDK's
    # Core and Shaders to the browser -- the server never renders a Cubism
    # model. `live2d_enabled` hides the whole feature from the console.
    live2d_enabled: bool = True
    live2d_sdk_dir: Path = Path("data/CubismSdkForWeb-5-r.5")

    model_config = SettingsConfigDict(env_prefix="DH_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
