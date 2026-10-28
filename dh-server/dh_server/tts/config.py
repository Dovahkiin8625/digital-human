"""MiniMax TTS configuration: DB rows over built-in defaults.

The MiniMax settings live in the existing ``settings`` table (one row per
``minimax_*`` key). The admin console writes here via
``PUT /api/admin/tts/config``; the synthesise path reads here on every
call so a console edit takes effect on the very next request without a
server restart.

There is deliberately **no env layer**: the Models tab's dialog is the
single entry point -- the same shape the LLM uses (see
``dh_server/llm/config.py``). The machinery they share lives in
``dh_server/service_config.py``.
"""
from __future__ import annotations

from typing import Any

from dh_server.service_config import (
    ServiceConfig,
    mask_api_key,  # re-exported: the API surface imports it from here
    validate_api_key,
    validate_base_url,
)

#: The five configurable keys. ``api_key`` is the bearer token;
#: ``group_id`` is optional (newer MiniMax accounts skip it);
#: ``base_url`` lets an operator point at a self-hosted or staging
#: proxy; ``model`` is the speech synthesis model id; ``sample_rate``
#: is the output PCM rate (MiniMax caps at 44100 Hz). ``voice_id`` is
#: not a server-wide setting: the operator picks it per session in the
#: debug page.
KEYS: tuple[str, ...] = (
    "minimax_api_key",
    "minimax_group_id",
    "minimax_base_url",
    "minimax_model",
    "minimax_sample_rate",
)

#: Friendly aliases (used by the API endpoint body / GET response).
PUBLIC_KEYS: tuple[str, ...] = (
    "api_key",
    "group_id",
    "base_url",
    "model",
    "sample_rate",
)

#: Sample rates MiniMax accepts. Out-of-set values get rejected at write
#: time with a clear ``AppError`` -- ``11025`` and ``48000`` look plausible
#: but are not in the upstream ``audio_setting.sample_rate`` enumeration.
SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000, 32000, 44100)

#: Fallback for a key with no DB row. ``api_key`` / ``group_id`` are
#: empty on purpose: a fresh install has no credential, which the
#: synthesise path reports as 503 "not configured" rather than calling
#: out. The rest are the upstream defaults.
DEFAULTS: dict[str, Any] = {
    "minimax_api_key": "",
    "minimax_group_id": "",
    "minimax_base_url": "https://api.minimaxi.com",
    "minimax_model": "speech-02-hd",
    "minimax_sample_rate": 32000,
}


def _validate_sample_rate(value: Any) -> "str | None":
    """The rate has to be an integer MiniMax actually accepts."""
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return "must be an integer"
    if rate not in SUPPORTED_SAMPLE_RATES:
        return f"must be one of {list(SUPPORTED_SAMPLE_RATES)}"
    return None


_config = ServiceConfig(
    keys=KEYS,
    public_keys=PUBLIC_KEYS,
    defaults=DEFAULTS,
    error_code="invalid_minimax_config",
    error_message="one or more MiniMax fields are invalid",
    validators={
        "sample_rate": _validate_sample_rate,
        "base_url": validate_base_url,
        "api_key": validate_api_key,
        # model / group_id: free-form strings. No validation.
    },
)


def read_config(db) -> dict[str, Any]:
    """The currently effective MiniMax config, in the public shape."""
    return _config.read(db)


def validate(payload: dict[str, Any]) -> dict[str, str]:
    """Return ``{public_key: msg}`` errors for each invalid field."""
    return _config.validate(payload)


def write_config(db, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate ``payload``, persist each key, return the effective config."""
    return _config.write(db, payload)
