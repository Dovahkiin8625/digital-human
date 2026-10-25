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
    (
        # Which digital-human model drives the avatar. Replaces the old
        # ``kind`` column, whose two values ("photoreal" / "live2d") were
        # really two drive models wearing a different name.
        "avatars",
        "drive_model",
        "ALTER TABLE avatars ADD COLUMN drive_model VARCHAR(32) NOT NULL DEFAULT 'musetalk'",
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


def _rebuild_table_without_columns(
    engine, table: str, drop: set[str], keep: dict[str, str]
) -> None:
    """Recreate ``table`` without the named columns, copying data over.

    SQLite's ``ALTER TABLE ... DROP COLUMN`` can't drop a column that's
    the source of a foreign key — it errors with "unknown column in
    foreign key definition" because the FK metadata is baked into the
    table schema. The portable workaround is the classic SQLite dance:

        CREATE TABLE _new LIKE old (without the unwanted column);
        INSERT INTO _new SELECT <keep cols> FROM old;
        DROP TABLE old;
        ALTER TABLE _new RENAME TO old;

    FK checks must be off for the duration or the DROP of the old
    table violates the FKs that the new table is about to lose.

    ``keep`` maps the destination column name to its SQL type (used to
    build the ``CREATE TABLE`` statement). ``drop`` lists columns from
    the source schema that should not be carried over.
    """
    inspector = inspect(engine)
    src_cols = [c["name"] for c in inspector.get_columns(table)]
    carry_over = [c for c in src_cols if c not in drop]
    if set(carry_over) != set(keep):
        raise RuntimeError(
            f"{table}: rebuild keep/drop mismatch — keep={sorted(keep)}, "
            f"src carry-over={sorted(carry_over)}"
        )

    col_list = ", ".join(carry_over)
    new_cols = ", ".join(f"{name} {decl}" for name, decl in keep.items())
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(text(f"CREATE TABLE _{table} ({new_cols})"))
        conn.execute(
            text(f"INSERT INTO _{table} ({col_list}) SELECT {col_list} FROM {table}")
        )
        conn.execute(text(f"DROP TABLE {table}"))
        conn.execute(text(f"ALTER TABLE _{table} RENAME TO {table}"))
        conn.execute(text("PRAGMA foreign_keys = ON"))


def _backfill_drive_model(engine) -> None:
    """Carry the old ``kind`` column over into ``drive_model``.

    Before the drive-model binding, an avatar's ``kind`` said which
    renderer it used: ``photoreal`` (a MuseTalk pack) or ``live2d`` (a
    Cubism project). Those two values are really two *drive models*, so
    the rename is a straight mapping for every row that exists.

    Runs once, before the column is dropped. A database created with the
    current schema has no ``kind`` column and is left alone.
    """
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("avatars")}
    if "kind" not in cols or "drive_model" not in cols:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE avatars SET drive_model = 'live2d' "
                "WHERE kind = 'live2d' AND drive_model <> 'live2d'"
            )
        )


def _drop_abandoned_columns(engine) -> None:
    """Remove columns the current models no longer carry.

    The original service was multi-tenant: a ``tenants`` table plus
    ``tenant_id`` NOT NULL columns on ``avatars`` and ``sessions``. A
    later refactor dropped the ``Tenant`` model and its columns from
    ``server.db.models`` but not from existing DBs — SQLite's
    ``create_all`` only adds tables, so the stale schema stayed. Inserts
    into ``sessions`` then failed with ``NOT NULL constraint failed:
    sessions.tenant_id`` because the new model no longer populates that
    column.

    The same thing happened to ``avatars.kind`` when the drive-model
    binding replaced it with ``avatars.drive_model``: SQLite cannot
    ``ALTER TABLE ... DROP COLUMN`` a column that a foreign key refers
    to, so the table is rebuilt from a whitelist instead.

    This brings the DB in line with the current models.
    Idempotent: each step is gated by the target's presence, so re-runs
    and fresh DBs are no-ops.
    """
    inspector = inspect(engine)
    avatars_cols = {c["name"]: c["type"].compile() for c in inspector.get_columns("avatars")}
    sessions_cols = {c["name"]: c["type"].compile() for c in inspector.get_columns("sessions")}

    if "tenant_id" in avatars_cols or "kind" in avatars_cols:
        # Mirror the current Avatar model's column types so the rebuilt
        # table matches what ``Base.metadata.create_all`` would produce.
        keep = {name: avatars_cols[name] for name in (
            "id", "name", "status", "source_image_path", "pack_path",
            "config_json", "tunable_overrides", "drive_model", "created_at",
        )}
        _rebuild_table_without_columns(engine, "avatars", {"tenant_id", "kind"}, keep)
        inspector = inspect(engine)
    if "tenant_id" in {c["name"] for c in inspector.get_columns("sessions")}:
        keep = {name: sessions_cols[name] for name in (
            "id", "avatar_id", "status", "started_at", "closed_at", "metrics_json",
        )}
        _rebuild_table_without_columns(engine, "sessions", {"tenant_id"}, keep)
        inspector = inspect(engine)
    if "tenants" in inspector.get_table_names():
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE tenants"))


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    _apply_migrations(engine)
    # The backfill reads the column the rebuild is about to drop, so it
    # has to run in between.
    _backfill_drive_model(engine)
    _drop_abandoned_columns(engine)


def get_db() -> Iterator[Session]:
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
