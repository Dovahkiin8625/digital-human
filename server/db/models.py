import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Avatar(Base):
    __tablename__ = "avatars"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    #: Which renderer this avatar is for. ``photoreal`` rows go through the
    #: MuseTalk / LivePortrait / Wan pack pipeline; ``live2d`` rows are a
    #: Cubism project the client renders itself. A third renderer means a
    #: third value here, not a second table.
    kind: Mapped[str] = mapped_column(String(16), default="photoreal", nullable=False)
    source_image_path: Mapped[str] = mapped_column(String(512))
    pack_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # Per-avatar tunable overrides captured by the create form (idle_crf,
    # idle_preset). The pipeline and the idle-video rebuild path both read
    # this dict through ``tunables.effective(db, overrides=...)`` so a
    # per-avatar choice survives the build and a later re-encode. Empty
    # by default; the column is NOT NULL so the pipeline can pass the
    # value through without a None-check.
    tunable_overrides: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AvatarJob(Base):
    __tablename__ = "avatar_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(32), ForeignKey("avatars.id"))
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class StreamSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(String(32), ForeignKey("avatars.id"))
    status: Mapped[str] = mapped_column(String(16), default="idle")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Setting(Base):
    """Admin-controlled tunables stored as key/value JSON.

    The admin console writes via ``PATCH /api/admin/settings``; the avatar
    pipeline and idle-video encoder read through
    ``server.avatar.tunables.effective(db)`` on every call so a runtime change
    takes effect on the next avatar without a server restart.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(1024))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class GenerationTask(Base):
    """Unified task row for async, long-running API operations.

    Currently populated by the ``/api/admin/i2v`` endpoint (image-to-video).
    ``type`` discriminates the row so future long ops (e.g. batch avatar
    re-encode) reuse the same table without a new schema. Lifecycle:

        queued   -> running   -> done | failed

    The frontend polls ``/api/admin/tasks/{id}`` until the row reaches a
    terminal state, then reads the result mp4 from
    ``/api/admin/tasks/{id}/result.mp4``.

    ``params_json`` / ``result_json`` are JSON-encoded strings: SQLite has no
    native JSON column on the venv in use here, and the server already uses
    the same pattern for ``Setting.value``. They round-trip through
    ``json.loads`` on the API side so the JSON shape stays ergonomic.
    """

    __tablename__ = "generation_tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    # User-visible failure message; NULL while running or on success.
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Free-form input payload (JSON-encoded). For i2v this is the same shape
    # the form posted -- backend name + knobs + the original filename hint.
    params_json: Mapped[str] = mapped_column(String(4096), default="{}")
    # Free-form result payload. For i2v the shape is ``{"mp4_path": "..."}``
    # so the result endpoint can stream the bytes without re-running work.
    result_json: Mapped[str | None] = mapped_column(String(4096), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)