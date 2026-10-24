"""MiniMax TTS configuration: DB rows over built-in defaults.

The MiniMax settings live in the existing ``settings`` table (one row per
``minimax_*`` key). The admin console writes here via
``PUT /api/admin/tts/config``; the synthesise path reads here on every
call so a console edit takes effect on the very next request without a
server restart.

There is deliberately **no env layer**: the ``DH_MINIMAX_*`` vars are
gone, and the Models tab's dialog is the single entry point -- the same
shape the LLM uses (see ``server/llm/config.py``). Two persistence
layers for one credential meant the Models tab's "已配置" badge tracked
the env copy while the synthesise path used the DB copy, so an operator
who saved a key in the console still saw the service as unconfigured.
"""
from __future__ import annotations

import json
from typing import Any

#: The five configurable keys. ``api_key`` is the bearer token;
#: ``group_id`` is optional (newer MiniMax accounts skip it);
#: ``base_url`` lets an operator point at a self-hosted or staging
#: proxy; ``model`` is the speech synthesis model id; ``sample_rate``
#: is the output PCM rate (MiniMax caps at 44100 Hz). ``voice_id``
#: used to live here too -- the operator now picks it per session in
#: the debug page, so it isn't a server-wide setting anymore.
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

#: Wire name ↔ public name. ``_KEY_TO_PUBLIC[k]`` is the key the API
#: surface exposes; ``_PUBLIC_TO_KEY[p]`` is the DB key.
_KEY_TO_PUBLIC = dict(zip(KEYS, PUBLIC_KEYS))
_PUBLIC_TO_KEY = dict(zip(PUBLIC_KEYS, KEYS))

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


def _public_view(values: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``minimax_`` prefix so the API matches what the form sends."""
    return {_KEY_TO_PUBLIC[k]: values[k] for k in KEYS}


def read_config(db) -> dict[str, Any]:
    """Return the *currently effective* MiniMax config for one DB session.

    DB rows win over :data:`DEFAULTS` on a per-key basis; absent keys
    fall back to the built-in default. Returns the public (no-prefix)
    key shape so the API surface and the synthesise path agree.
    """
    from sqlalchemy import select

    from server.db.models import Setting

    out = dict(DEFAULTS)
    rows = db.execute(
        select(Setting).where(Setting.key.in_(KEYS))
    ).scalars().all()
    for row in rows:
        try:
            out[row.key] = json.loads(row.value)
        except (ValueError, TypeError):
            # A malformed value falls back to the default rather than
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
        if public_name == "sample_rate":
            try:
                v = int(value)
            except (TypeError, ValueError):
                errors[public_name] = "must be an integer"
                continue
            if v not in SUPPORTED_SAMPLE_RATES:
                errors[public_name] = (
                    f"must be one of {list(SUPPORTED_SAMPLE_RATES)}"
                )
        elif public_name == "base_url":
            if not isinstance(value, str) or not value.startswith(("http://", "https://")):
                errors[public_name] = "must start with http:// or https://"
        elif public_name == "api_key":
            # Empty api_key means "not configured"; we accept it but the
            # synthesise endpoint will then return 503. We don't refuse
            # a clear-then-save flow.
            if value is not None and not isinstance(value, str):
                errors[public_name] = "must be a string"
        # model / group_id: free-form strings. No validation.
    return errors


def write_config(db, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate ``payload``, persist each key, return the effective config.

    Each non-empty public field maps to a DB row keyed by the wire
    ``minimax_*`` name. The function then re-reads via :func:`read_config`
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
            "invalid_minimax_config",
            "one or more MiniMax fields are invalid",
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