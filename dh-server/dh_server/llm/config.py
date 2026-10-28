"""LLM configuration: DB row overrides on top of built-in defaults.

There is deliberately **no env layer**: the admin console writes per-key
values to the ``settings`` table via ``PUT /api/admin/llm/config``; the
chat path reads the layered result on every call so a console edit takes
effect on the very next request without a server restart.

The MiniMax TTS config (``dh_server/tts/config.py``) has the same shape --
DB rows over built-in defaults, configured from the Models tab. The
machinery they share lives in ``dh_server/service_config.py``.

Three configurable keys:

* ``llm_api_key`` -- bearer token. Empty means "not configured"; the chat
  endpoint returns 503 rather than calling out.
* ``llm_base_url`` -- upstream root. The chat path appends
  ``/chat/completions``; ``https://api.openai.com/v1`` for OpenAI,
  ``http://localhost:11434/v1`` for Ollama, etc.
* ``llm_model`` -- the upstream model identifier the operator wants the
  console pointed at. Sent on every request so the same console can swap
  providers without code.
"""
from __future__ import annotations

from typing import Any

from dh_server.service_config import (
    ServiceConfig,
    mask_api_key,  # re-exported: the API surface imports it from here
    validate_api_key,
    validate_base_url,
    validate_model,
)

KEYS: tuple[str, ...] = (
    "llm_api_key",
    "llm_base_url",
    "llm_model",
)

#: Friendly aliases (used by the API endpoint body / GET response).
PUBLIC_KEYS: tuple[str, ...] = (
    "api_key",
    "base_url",
    "model",
)

#: Fallback for a key with no DB row. ``api_key`` / ``base_url`` are
#: empty on purpose: a fresh install has no credential, which the chat
#: path reports as 503 "not configured" rather than calling out.
DEFAULTS: dict[str, Any] = {
    "llm_api_key": "",
    "llm_base_url": "",
    "llm_model": "gpt-3.5-turbo",
}

_config = ServiceConfig(
    keys=KEYS,
    public_keys=PUBLIC_KEYS,
    defaults=DEFAULTS,
    error_code="invalid_llm_config",
    error_message="one or more LLM fields are invalid",
    validators={
        "base_url": validate_base_url,
        "api_key": validate_api_key,
        "model": validate_model,
    },
)


def read_config(db) -> dict[str, Any]:
    """The currently effective LLM config, in the public shape."""
    return _config.read(db)


def validate(payload: dict[str, Any]) -> dict[str, str]:
    """Return ``{public_key: msg}`` errors for each invalid field."""
    return _config.validate(payload)


def write_config(db, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate ``payload``, persist each key, return the effective config."""
    return _config.write(db, payload)


def is_configured(cfg: dict[str, Any]) -> bool:
    """True iff the operator has wired an api_key AND a base_url.

    The model field always has a default (``gpt-3.5-turbo``), so an unset
    model is never the blocker; missing api_key or base_url is.
    """
    return bool(cfg.get("api_key")) and bool(cfg.get("base_url"))
