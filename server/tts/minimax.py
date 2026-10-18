"""MiniMax text-to-speech, for the console's text-driven debug page.

The render session already ingests 48 kHz s16le PCM over its audio WebSocket;
this module turns typed text into bytes that channel accepts. MiniMax returns
the audio as a hex string and never at 48000 Hz (its ceiling is 44100), so we
ask for raw PCM at the configured rate, wrap it in a WAV container, and let the
browser decode + resample with the helper the file-upload path already uses.

Keeping the resample in the browser is deliberate: it means no audio codec on
the server, and the WAV header only has to describe the bytes that follow.
"""

import struct

import httpx

from server.config import get_settings
from server.core.errors import AppError

_WAV_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")


def wav_from_pcm(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap 16-bit PCM in a canonical 44-byte WAV header.

    The header is fixed-size so the payload always starts at byte 44, which the
    tests and the browser's decoder both rely on.
    """
    bits = 16
    block_align = channels * bits // 8
    header = _WAV_HEADER.pack(
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk length
        1,  # format: PCM
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
        b"data",
        len(pcm),
    )
    return header + pcm


def _require_configured():
    """Legacy stub kept for callers that pass ``db=None`` (the test path).

    Real callers should rely on :func:`server.tts.config.read_config`'s
    own "not configured" branch in :func:`synthesize` -- this helper only
    exists so importing the symbol doesn't break older call sites.
    """
    from server.config import get_settings

    settings = get_settings()
    if not settings.minimax_api_key:
        raise AppError(
            "tts_not_configured",
            "MiniMax is not configured: set DH_MINIMAX_API_KEY in .env",
            503,
        )
    return settings


def _audio_from_response(body: dict, status_code: int) -> str:
    """Pull the hex audio out of a MiniMax reply, or raise a 502 that says why."""
    base = body.get("base_resp") or {}
    if base.get("status_code"):
        raise AppError(
            "tts_failed",
            f"MiniMax error {base['status_code']}: {base.get('status_msg', '')}".strip(),
            502,
        )
    audio = (body.get("data") or {}).get("audio") or ""
    if not audio:
        raise AppError("tts_failed", f"MiniMax returned no audio (HTTP {status_code})", 502)
    return audio


async def synthesize(text: str, voice_id: str | None = None, *, client: httpx.AsyncClient | None = None, db=None) -> bytes:
    """Synthesize ``text`` into WAV bytes, or raise AppError.

    ``client`` is injectable so tests can drive the real parse path through an
    httpx.MockTransport instead of the network. ``db`` is the FastAPI request
    session; passed through so the effective MiniMax config (DB rows layered
    over env defaults) is read fresh on every call, not cached.
    """
    # Read from the layered config helper rather than get_settings() so a
    # console edit via ``PUT /api/admin/tts/config`` takes effect on the
    # very next request without a server restart. Falls back to env
    # defaults when ``db`` is None (test path).
    from server.tts.config import read_config

    if db is not None:
        cfg = read_config(db)
    else:
        cfg = _settings_defaults_as_public()

    if not cfg.get("api_key"):
        raise AppError(
            "tts_not_configured",
            "MiniMax is not configured: set DH_MINIMAX_API_KEY in .env or save it via the Models tab",
            503,
        )

    payload = {
        "model": cfg["model"],
        "text": text,
        "stream": False,
        # "hex" is the only output format streaming supports and the one the
        # non-streaming default returns, so the parse path never branches.
        "output_format": "hex",
        "voice_setting": {"voice_id": voice_id or cfg["voice_id"], "speed": 1, "vol": 1, "pitch": 0},
        "audio_setting": {"format": "pcm", "sample_rate": cfg["sample_rate"], "channel": 1},
    }
    params = {"GroupId": cfg["group_id"]} if cfg["group_id"] else None
    url = f"{cfg['base_url']}/v1/t2a_v2"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=60)
    try:
        try:
            resp = await client.post(url, params=params, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise AppError("tts_failed", f"MiniMax request failed: {exc}", 502) from exc
        try:
            body = resp.json()
        except ValueError as exc:
            raise AppError(
                "tts_failed", f"MiniMax returned a non-JSON response (HTTP {resp.status_code})", 502
            ) from exc
    finally:
        if owns_client:
            await client.aclose()

    hex_audio = _audio_from_response(body, resp.status_code)
    try:
        pcm = bytes.fromhex(hex_audio)
    except ValueError as exc:
        # output_format=hex is pinned, so this means the reply is not what we
        # asked for (e.g. a URL) -- name it rather than 500 on a bare ValueError.
        raise AppError("tts_failed", "MiniMax returned audio that is not hex", 502) from exc
    return wav_from_pcm(pcm, cfg["sample_rate"])


def _settings_defaults_as_public() -> dict:
    """Mirror of ``read_config`` for callers without a DB session.

    The test path (and any future non-HTTP caller) needs the same
    public-shape dict without going through SQLAlchemy.
    """
    from server.config import get_settings

    s = get_settings()
    return {
        "api_key": getattr(s, "minimax_api_key", ""),
        "group_id": getattr(s, "minimax_group_id", ""),
        "base_url": getattr(s, "minimax_base_url", "https://api.minimaxi.com"),
        "model": getattr(s, "minimax_model", "speech-02-hd"),
        "voice_id": getattr(s, "minimax_voice_id", "male-qn-qingse"),
        "sample_rate": getattr(s, "minimax_sample_rate", 32000),
    }
