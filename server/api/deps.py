import hashlib
import hmac

from fastapi import Depends, Header

from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Tenant
from server.db.session import get_db


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_tenant(x_api_key: str = Header(default=""), db=Depends(get_db)) -> Tenant:
    if not x_api_key:
        raise AppError("unauthorized", "missing or invalid API key", 401)
    tenant = db.query(Tenant).filter(Tenant.api_key_hash == hash_api_key(x_api_key)).first()
    if tenant is None:
        raise AppError("unauthorized", "missing or invalid API key", 401)
    return tenant


def get_admin(x_admin_token: str = Header(default="")) -> None:
    expected = get_settings().admin_token
    if not hmac.compare_digest(x_admin_token.encode("utf-8"), expected.encode("utf-8")):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
