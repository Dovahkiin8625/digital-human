"""Credentials for the client-facing endpoints.

Only the business-plane surface is gated, and only when a token is
configured. The existing ``/api/v1`` routes are unauthenticated after the
no-tenant refactor and stay that way here: retrofitting them would churn
every current caller, and it is a separate piece of work with its own
blast radius.

An unset token means open. That matches how every admin endpoint already
behaves on a single-operator box, and it means the admin console -- which
carries no credential of its own -- keeps working without changes. A
deployment that puts this service on a reachable network sets a token.
"""

import secrets

from fastapi import Header

from server.config import get_settings
from server.core.errors import AppError


async def require_v1_token(x_v1_token: str | None = Header(default=None)) -> None:
    """Pass when no token is configured, or when the header matches.

    ``compare_digest`` rather than ``==`` so the comparison does not leak
    the token's length or a matching prefix through response timing.
    """
    expected = get_settings().v1_api_token
    if not expected:
        return
    if not x_v1_token or not secrets.compare_digest(x_v1_token, expected):
        raise AppError("unauthorized", "missing or invalid X-V1-Token", 401)