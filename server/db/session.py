from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from server.config import get_settings
from server.db.models import Base

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    return _engine


def reset_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


#: Additive ALTER TABLE statements run on every init_db(). SQLite doesn't
#: support ``IF NOT EXISTS`` on ADD COLUMN, so each statement has to be
#: issued inside an idempotency guard (the ``has_column`` check below).
#: Removing entries from this list is fine; the column is then simply
#: left in place on upgraded databases.
_SCHEMA_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    # (table, column, ddl-fragment-for-existing-table)
    (
        "avatars",
        "tunable_overrides",
        "ALTER TABLE avatars ADD COLUMN tunable_overrides JSON NOT NULL DEFAULT '{}'",
    ),
)


def _apply_migrations(engine) -> None:
    """Apply additive ALTER TABLEs idempotently.

    Runs once per ``init_db``. Each migration is skipped when the target
    column already exists, so re-running on a fresh DB or after a prior
    successful run is a no-op.
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl in _SCHEMA_MIGRATIONS:
            if column in {c["name"] for c in inspector.get_columns(table)}:
                continue
            conn.execute(text(ddl))


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    _apply_migrations(engine)


def get_db() -> Iterator[Session]:
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
