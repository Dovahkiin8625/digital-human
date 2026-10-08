from fastapi import FastAPI

from server.config import get_settings
from server.db.session import get_engine, init_db


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="digital-human", version="0.1.0")
    init_db(get_engine())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "engine": settings.drive_engine, "gpu": False}

    return app
