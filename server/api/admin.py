import hmac

from fastapi import APIRouter
from pydantic import BaseModel

from server.config import get_settings
from server.core.errors import AppError

router = APIRouter(prefix="/api/admin", tags=["admin"])


class LoginIn(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginIn) -> dict:
    if not hmac.compare_digest(body.password.encode("utf-8"), get_settings().admin_password.encode("utf-8")):
        raise AppError("unauthorized", "missing or invalid admin token", 401)
    return {"token": get_settings().admin_token}
