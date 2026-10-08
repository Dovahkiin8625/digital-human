from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


def envelope(code: str, message: str, detail: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail}}


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content=envelope(exc.code, exc.message, exc.detail))

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(status_code=exc.status_code, content=envelope(code, str(exc.detail), None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content=envelope("validation_error", "invalid request", exc.errors()))

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content=envelope("internal_error", "internal server error", str(exc)))
