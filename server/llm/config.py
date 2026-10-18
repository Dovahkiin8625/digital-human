"""LLM configuration: env defaults + DB row overrides.

Mirrors the layer shape ``server/tts/config.py`` uses for MiniMax: env
defaults come from ``server.config.Settings`` (the ``DH_LLM_*`` group); the
admin console writes per-key overrides to the ``settings`` table via
``PUT /api/admin/llm/config``; the chat path reads the layered result on
every call so a console edit takes effect on the very next request without
a server restart.

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

import json
from typing import Any

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

#: Wire name ↔ public name. ``_KEY_TO_PUBLIC[k]`` is the key the API
#: surface exposes; ``_PUBLIC_TO_KEY[p]`` is the DB / env key.
_KEY_TO_PUBLIC = dict(zip(KEYS, PUBLIC_KEYS))
_PUBLIC_TO_KEY = dict(zip(PUBLIC_KEYS, KEYS))


def _env_defaults() -> dict[str, Any]:
    """Default values when no DB row is present.

    Deliberately all empty: the LLM is configured via the Models tab
    only (see ``PUT /api/admin/llm/config``). No env-var fallback on
    purpose -- a freshly-deployed box should show the LLM row as
    "未配置" until an operator wires it through the UI, the same way
    MiniMax TTS surfaces as not-configured until an api_key lands.
    """
    return {
        "llm_api_key": "",
        "llm_base_url": "",
        "llm_model": "gpt-3.5-turbo",
    }


def _public_view(env: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``llm_`` prefix so the API matches what the form sends."""
    return {_KEY_TO_PUBLIC[k]: env[k] for k in KEYS}


def read_config(db) -> dict[str, Any]:
    """Return the *currently effective* LLM config for one DB session.

    DB rows win over env defaults on a per-key basis; absent keys fall
    back to whatever ``Settings`` returned. Returns the public (no-prefix)
    key shape so the API surface and the chat path agree.
    """
    from sqlalchemy import select

    from server.db.models import Setting

    out = _env_defaults()
    rows = db.execute(
        select(Setting).where(Setting.key.in_(KEYS))
    ).scalars().all()
    for row in rows:
        try:
            out[row.key] = json.loads(row.value)
        except (ValueError, TypeError):
            # A malformed value falls back to the env default rather than
            # 500-ing the caller; the operator can re-PUT to fix it.
            continue
    return _public_view(out)


def validate(payload: dict[str, Any]) -> dict[str, str]:
    """Return ``{public_key: msg}`` errors for each invalid field.

    Empty dict means OK. Unknown keys (typos in the operator's POST) get
    flagged so they don't silently land in the DB.
    """
    errors: dict[str, str] = {}
    for public_name, value in payload.items():
        if public_name not in _PUBLIC_TO_KEY:
            errors[public_name] = "unknown field"
            continue
        if public_name == "base_url":
            if not isinstance(value, str) or not value.startswith(("http://", "https://")):
                errors[public_name] = "must start with http:// or https://"
        elif public_name == "api_key":
            # Empty api_key means "not configured"; we accept it but the
            # chat endpoint will then return 503. We don't refuse a
            # clear-then-save flow.
            if value is not None and not isinstance(value, str):
                errors[public_name] = "must be a string"
        elif public_name == "model":
            if not isinstance(value, str) or not value.strip():
                errors[public_name] = "must be a non-empty string"
    return errors


def write_config(db, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate ``payload``, persist each key, return the effective config.

    Each non-empty public field maps to a DB row keyed by the wire
    ``llm_*`` name. The function then re-reads via :func:`read_config`
    so the caller sees the post-write view (including any fields they
    didn't touch).

    Raises :class:`server.core.errors.AppError` with a ``detail`` mapping
    of per-field messages so the API layer returns a 400 the UI can
    surface inline.
    """
    from server.core.errors import AppError

    errors = validate(payload)
    if errors:
        raise AppError(
            "invalid_llm_config",
            "one or more LLM fields are invalid",
            400,
            detail=errors,
        )

    from server.db.models import Setting

    for public_name, value in payload.items():
        wire_key = _PUBLIC_TO_KEY[public_name]
        existing = db.get(Setting, wire_key)
        if existing is None:
            existing = Setting(key=wire_key, value=json.dumps(value))
            db.add(existing)
        else:
            existing.value = json.dumps(value)
    db.commit()

    return read_config(db)


def mask_api_key(cfg: dict[str, Any]) -> str:
    """Render ``api_key`` as ``****<last4>`` for the GET response.

    Empty keys render as ``"(未设置)"`` so the operator sees the
    not-configured state without us shipping an empty string they might
    mistake for "I have a key but it's blank".
    """
    api_key = cfg.get("api_key") or ""
    if not api_key:
        return "(未设置)"
    if len(api_key) <= 4:
        return "*" * len(api_key)
    return "****" + api_key[-4:]


def is_configured(cfg: dict[str, Any]) -> bool:
    """True iff the operator has wired an api_key AND a base_url.

    The model field always has a default (``gpt-3.5-turbo``), so an unset
    model is never the blocker; missing api_key or base_url is.
    """
    return bool(cfg.get("api_key")) and bool(cfg.get("base_url"))
