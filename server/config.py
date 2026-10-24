from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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

    # NOTE: the MiniMax TTS config (api_key / group_id / base_url / model /
    # sample_rate) lives entirely in the SQLite ``settings`` table and is
    # configured from the Models tab, exactly like the LLM below. There are
    # deliberately no ``DH_MINIMAX_*`` env defaults: two persistence layers
    # for one credential meant the console's badge tracked the wrong one.
    # MiniMax tops out at 44100 Hz, so the synthesised rate is never the
    # render session's 48000 -- the browser resamples onto the WS audio
    # channel (webrtc.ts).

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
    # model. `live2d_enabled` is *reported* (it lands in the Live2D health
    # payload the console's System tab renders) but not enforced: no
    # endpoint branches on it, so the Live2D upload route and the create
    # form's Live2D option keep working when it is False.
    live2d_enabled: bool = True
    live2d_sdk_dir: Path = Path("data/CubismSdkForWeb-5-r.5")
    # A Live2D upload is an operator-supplied zip of a whole Cubism project.
    # Three limits, because the archive size alone does not bound what an
    # upload does: a few hundred KB of zeros can expand to hundreds of MB,
    # and a small archive can declare a million members.
    live2d_max_archive_entries: int = 512
    live2d_max_unpacked_bytes: int = 512_000_000
    live2d_max_entry_bytes: int = 64_000_000

    # ``extra="ignore"`` rather than the pydantic-settings default of
    # ``"forbid"``: a box upgraded from a version that still shipped
    # ``DH_MINIMAX_*`` (or the retired ``DH_DRIVE_ENGINE`` /
    # ``DH_AVATAR_PIPELINE``) in its .env must not fail to start just
    # because those keys no longer have a Settings field. Unknown DH_*
    # vars are inert; they are simply not read by anything.
    #
    # Which digital-human model this box runs is NOT an env concern: it is
    # an operator choice stored in the SQLite ``settings`` table, the same
    # place the MiniMax TTS and LLM configs live. See
    # ``server/model_registry.py``.
    model_config = SettingsConfigDict(env_prefix="DH_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
