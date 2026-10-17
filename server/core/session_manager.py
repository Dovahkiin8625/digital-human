import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.avatar.store import AvatarStore
from server.config import get_settings
from server.core.errors import AppError
from server.core.render_session import RenderSession
from server.core.stream_sink import NullSink, StreamSink
from server.db.models import Avatar, StreamSession
from server.engine.base import DriveEngine


class SessionManager:
    def __init__(self, engine: DriveEngine, store: AvatarStore, idle_timeout: float = 60.0):
        self.engine = engine
        self.store = store
        self.idle_timeout = idle_timeout
        self._renders: dict[str, RenderSession] = {}
        self._engine_closed = False

    async def create(self, db: Session, avatar: Avatar) -> StreamSession:
        row = StreamSession(avatar_id=avatar.id, status="idle")
        db.add(row)
        db.commit()
        pack = self.store.load_pack(avatar)
        settings = get_settings()
        render = RenderSession(
            row.id,
            pack,
            self.engine,
            NullSink(),
            sample_rate=settings.sample_rate,
            fps=settings.fps,
        )
        await render.start()
        self._renders[row.id] = render
        return row

    def get_render(self, session_id: str) -> RenderSession:
        render = self._renders.get(session_id)
        if render is None:
            raise AppError("not_found", f"session {session_id} not found", 404)
        return render

    def attach_sink(self, session_id: str, sink: StreamSink) -> None:
        self.get_render(session_id).sink = sink

    async def close(self, db: Session, session_id: str) -> None:
        render = self._renders.pop(session_id, None)
        if render is not None:
            await render.close()
        row = db.get(StreamSession, session_id)
        if row is not None and row.status != "closed":
            row.status = "closed"
            row.closed_at = datetime.now(timezone.utc)
            if render is not None:
                row.metrics_json = render.metrics
            db.commit()

    async def sweep_idle(self, db: Session, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        closed = 0
        for sid, render in list(self._renders.items()):
            if now - render.last_audio_at > self.idle_timeout:
                await self.close(db, sid)
                closed += 1
        return closed

    async def shutdown(self) -> None:
        """Close the shared engine once. Idempotent and safe to call repeatedly.

        RenderSession.close() no longer touches the engine, so this is the single
        place the process-wide DriveEngine is torn down (wired to app lifespan
        shutdown). A second call is a no-op.
        """
        if self._engine_closed:
            return
        self._engine_closed = True
        # Sessions still open at teardown never reach close(), so their
        # keep-alive tasks would outlive the loop that owns them.
        for render in self._renders.values():
            render.stop()
        await self.engine.close()
