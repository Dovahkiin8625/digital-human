import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from server.config import get_settings
from server.db.session import get_engine, init_db
from server.engine.base import gpu_available

logger = logging.getLogger(__name__)

_app = None


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves the SPA index for client-side routes.

    A request for a real file that is missing (e.g. ``/admin/assets/x.js``)
    must still 404; only extension-less paths (client-side routes such as
    ``/admin/avatars``) fall back to ``index.html`` so a refresh or deep link
    under the ``/admin`` mount does not break.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or os.path.splitext(path)[1]:
                raise
            return await super().get_response("index.html", scope)


async def _sweep_loop(manager, interval: float, session_factory) -> None:
    """Periodically reclaim idle sessions until cancelled (spec §6: 60s auto-reclaim)."""
    while True:
        await asyncio.sleep(interval)
        # A transient failure in one tick (e.g. a broken tail-drive) must not kill
        # the sweeper permanently; absorb it, log it, and keep ticking.
        try:
            db = session_factory()
            try:
                await manager.sweep_idle(db)
            finally:
                db.close()
        except Exception:
            logger.exception("session sweep tick failed")


async def _warmup(engine) -> None:
    """Load the engine's weights off the request path (cold start is ~34 s).

    Failure is logged, not fatal: a box without a GPU -- or one that has not
    fetched the weights yet -- must still boot, and the first session will
    surface the real error through its own prepare().
    """
    try:
        await engine.warmup()
        logger.info("engine %s warmup complete", engine.name)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning(
            "engine %s warmup failed; the first session will pay the cost",
            engine.name,
            exc_info=True,
        )


def create_app() -> FastAPI:
    global _app
    settings = get_settings()

    # Before any RTCPeerConnection exists: encoders read the aiortc module
    # globals at construction, so doing this lazily on the first session would
    # miss that session's encoder. Idempotent and cheap.
    from server.core.webrtc import configure_h264_bitrate

    configure_h264_bitrate(settings.video_bitrate, settings.video_max_bitrate)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from sqlalchemy.orm import sessionmaker

        def session_factory():
            return sessionmaker(bind=get_engine())()

        sweep_task = asyncio.create_task(
            _sweep_loop(app.state.session_manager, settings.session_sweep_interval, session_factory)
        )
        app.state.sweep_task = sweep_task
        warmup_task = asyncio.create_task(_warmup(app.state.session_manager.engine))
        app.state.warmup_task = warmup_task
        try:
            yield
        finally:
            warmup_task.cancel()
            sweep_task.cancel()
            for task in (warmup_task, sweep_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # Shutdown closes the shared engine, which waits out any in-flight
            # warmup on its own worker thread before releasing the GPU.
            await app.state.session_manager.shutdown()

    app = FastAPI(title="digital-human", version="0.1.0", lifespan=lifespan)
    init_db(get_engine())
    # Probe once at startup rather than on every /healthz: importing torch is
    # the expensive part, and the answer cannot change without a restart.
    app.state.gpu = gpu_available()

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "engine": settings.drive_engine,
            "pipeline": settings.avatar_pipeline,
            "gpu": app.state.gpu,
        }

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

    from pathlib import Path

    dist = Path(__file__).resolve().parent.parent / "admin-ui" / "dist"
    if dist.exists():
        app.mount("/admin", SPAStaticFiles(directory=str(dist), html=True), name="admin")

    _app = app
    return app


def get_session_manager():
    return _app.state.session_manager
