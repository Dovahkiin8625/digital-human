import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.config import get_settings
from server.db.session import get_engine, init_db

_app = None


async def _sweep_loop(manager, interval: float, session_factory) -> None:
    """Periodically reclaim idle sessions until cancelled (spec §6: 60s auto-reclaim)."""
    while True:
        await asyncio.sleep(interval)
        db = session_factory()
        try:
            await manager.sweep_idle(db)
        finally:
            db.close()


def create_app() -> FastAPI:
    global _app
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from sqlalchemy.orm import sessionmaker

        def session_factory():
            return sessionmaker(bind=get_engine())()

        sweep_task = asyncio.create_task(
            _sweep_loop(app.state.session_manager, settings.session_sweep_interval, session_factory)
        )
        app.state.sweep_task = sweep_task
        try:
            yield
        finally:
            sweep_task.cancel()
            try:
                await sweep_task
            except asyncio.CancelledError:
                pass
            await app.state.session_manager.shutdown()

    app = FastAPI(title="digital-human", version="0.1.0", lifespan=lifespan)
    init_db(get_engine())

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "engine": settings.drive_engine, "gpu": False}

    from server.api.admin import router as admin_router
    from server.api.avatars import router as avatars_router
    from server.api.sessions import router as sessions_router
    from server.avatar.store import AvatarStore
    from server.core.errors import register_handlers
    from server.core.session_manager import SessionManager
    from server.engine.base import create_engine

    register_handlers(app)
    app.include_router(admin_router)
    app.include_router(avatars_router)
    app.include_router(sessions_router)
    app.state.session_manager = SessionManager(
        create_engine(), AvatarStore(settings.data_dir), settings.session_idle_timeout
    )
    _app = app
    return app


def get_session_manager():
    return _app.state.session_manager
