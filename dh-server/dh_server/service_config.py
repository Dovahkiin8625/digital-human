"""The settings of one service, stored as JSON rows in the ``settings`` table.

Two services use it: the MiniMax TTS and the OpenAI-compatible LLM. They
have the same shape -- a handful of keys, a built-in default for each, one
row per key, and a console dialog that writes them -- so the machinery
lives here once and each service keeps only what is specific to it: its
keys, its defaults and the fields it validates.

The rows are read on every call rather than cached, so a change the
operator saves in the console takes effect on the very next request
without a server restart.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

#: A field validator returns an error message, or None when the value is
#: acceptable.
Validator = Callable[[Any], "str | None"]


def validate_base_url(value: Any) -> "str | None":
    """A base URL has to be an absolute http(s) URL."""
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return "must start with http:// or https://"
    return None


def validate_api_key(value: Any) -> "str | None":
    """An empty api key means "not configured" and is accepted.

    The service reports that state itself (503 "not configured"), so a
    clear-then-save flow is not refused here.
    """
    if value is not None and not isinstance(value, str):
        return "must be a string"
    return None


def validate_model(value: Any) -> "str | None":
    """A model id has to be a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        return "must be a non-empty string"
    return None


class ServiceConfig:
    """One service's settings: the keys, their defaults and their rules.

    ``keys`` are the row names in the ``settings`` table (``llm_api_key``);
    ``public_keys`` are the names the API surface uses (``api_key``). The
    two line up one for one, in order.
    """

    def __init__(
        self,
        *,
        keys: tuple[str, ...],
        public_keys: tuple[str, ...],
        defaults: dict[str, Any],
        error_code: str,
        error_message: str,
        validators: dict[str, Validator] | None = None,
    ):
        if len(keys) != len(public_keys):
            raise ValueError("keys and public_keys must line up")
        self.keys = keys
        self.public_keys = public_keys
        self.defaults = dict(defaults)
        self.error_code = error_code
        self.error_message = error_message
        self.validators = dict(validators or {})
        self._key_to_public = dict(zip(keys, public_keys))
        self._public_to_key = dict(zip(public_keys, keys))

    def _public_view(self, values: dict[str, Any]) -> dict[str, Any]:
        """Strip the row-name prefix so the API matches what the form sends."""
        return {self._key_to_public[k]: values[k] for k in self.keys}

    def read(self, db) -> dict[str, Any]:
        """The currently effective config, in the public shape.

        DB rows win over the built-in defaults on a per-key basis; absent
        keys fall back to the default.
        """
        from sqlalchemy import select

        from dh_server.db.models import Setting

        out = dict(self.defaults)
        rows = db.execute(
            select(Setting).where(Setting.key.in_(self.keys))
        ).scalars().all()
        for row in rows:
            try:
                out[row.key] = json.loads(row.value)
            except (ValueError, TypeError):
                # A malformed value falls back to the default rather than
                # 500-ing the caller; the operator can re-save to fix it.
                continue
        return self._public_view(out)

    def validate(self, payload: dict[str, Any]) -> dict[str, str]:
        """Return ``{public_key: msg}`` errors for each invalid field.

        Empty dict means OK. Unknown keys (typos in the operator's POST)
        get flagged so they don't silently land in the DB.
        """
        errors: dict[str, str] = {}
        for public_name, value in payload.items():
            if public_name not in self._public_to_key:
                errors[public_name] = "unknown field"
                continue
            check = self.validators.get(public_name)
            if check is None:
                continue
            message = check(value)
            if message:
                errors[public_name] = message
        return errors

    def write(self, db, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate ``payload``, persist each key, return the effective config.

        Raises :class:`dh_server.core.errors.AppError` with a ``detail``
        mapping of per-field messages so the API layer returns a 400 the
        UI can surface inline.
        """
        from dh_server.core.errors import AppError
        from dh_server.db.models import Setting

        errors = self.validate(payload)
        if errors:
            raise AppError(self.error_code, self.error_message, 400, detail=errors)

        for public_name, value in payload.items():
            wire_key = self._public_to_key[public_name]
            existing = db.get(Setting, wire_key)
            if existing is None:
                existing = Setting(key=wire_key, value=json.dumps(value))
                db.add(existing)
            else:
                existing.value = json.dumps(value)
        db.commit()

        return self.read(db)


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
