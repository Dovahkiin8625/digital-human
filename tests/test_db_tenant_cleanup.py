"""Cleanup of the abandoned multi-tenant DB schema.

The original service was multi-tenant (Avatar.tenant_id, StreamSession.tenant_id,
a ``tenants`` table). A later refactor dropped the Tenant model from
``server.db.models`` but did not migrate the DB: SQLite's ``create_all``
only adds new tables, so existing tenant columns stuck around. New
``INSERT INTO sessions`` then failed with ``NOT NULL constraint failed:
sessions.tenant_id`` because the model no longer populated that column.

``init_db`` must clean those artifacts up so the model and the schema
agree. The cleanup is idempotent so re-running on a fresh DB or after a
prior run is a no-op.
"""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, MetaData, String, Table, create_engine, inspect, text


def _build_old_schema(engine) -> None:
    """Hand-build the pre-no-tenant-refactor schema."""
    meta = MetaData()
    Table(
        "tenants",
        meta,
        Column("id", String(32), primary_key=True),
        Column("name", String(128)),
        Column("api_key_hash", String(64)),
        Column("created_at", String(32)),
    )
    Table(
        "avatars",
        meta,
        Column("id", String(32), primary_key=True),
        Column("tenant_id", String(32), ForeignKey("tenants.id"), nullable=False),
        Column("name", String(128)),
        Column("status", String(16)),
        Column("source_image_path", String(512)),
        Column("pack_path", String(512), nullable=True),
        Column("config_json", String(64)),
        Column("created_at", String(32)),
    )
    Table(
        "sessions",
        meta,
        Column("id", String(32), primary_key=True),
        Column("tenant_id", String(32), ForeignKey("tenants.id"), nullable=False),
        Column("avatar_id", String(32)),
        Column("status", String(16)),
        Column("started_at", String(32)),
        Column("closed_at", String(32), nullable=True),
        Column("metrics_json", String(64)),
    )
    meta.create_all(engine)


def test_init_db_drops_tenants_table(tmp_path):
    from server.db.session import init_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    _build_old_schema(engine)
    assert "tenants" in inspect(engine).get_table_names()

    init_db(engine)

    assert "tenants" not in inspect(engine).get_table_names()


def test_init_db_drops_tenant_id_from_avatars_and_sessions(tmp_path):
    from server.db.session import init_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    _build_old_schema(engine)

    init_db(engine)

    insp = inspect(engine)
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("avatars")}
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("sessions")}


def test_init_db_after_cleanup_allows_session_insert(tmp_path):
    """Round-trip a StreamSession through the ORM after cleanup.

    This is the exact insert that broke in production: the StreamSession
    model has no tenant_id, so a CREATE TABLE without that column is the
    only state where INSERT succeeds.
    """
    from server.db.session import init_db
    from sqlalchemy.orm import sessionmaker

    from server.db.models import StreamSession

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    _build_old_schema(engine)
    init_db(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.add(StreamSession(id="s" * 32, avatar_id="a" * 32, status="idle"))
        session.commit()
        assert session.get(StreamSession, "s" * 32) is not None
    finally:
        session.close()


def test_init_db_cleanup_is_idempotent(tmp_path):
    """Running init_db on an already-clean DB must not raise."""
    from server.db.session import init_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    init_db(engine)
    init_db(engine)  # second call: tenants/tenant_id already absent, must be a no-op

    insp = inspect(engine)
    assert "tenants" not in insp.get_table_names()
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("avatars")}
    assert "tenant_id" not in {c["name"] for c in insp.get_columns("sessions")}


def test_init_db_cleanup_preserves_existing_data(tmp_path):
    """The cleanup must drop tenant columns without touching avatar rows."""
    from server.db.session import init_db
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    _build_old_schema(engine)
    # Seed a tenant + avatar so we can prove the avatar row survives the cleanup.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tenants (id, name, api_key_hash, created_at) "
            "VALUES (:id, :name, :hash, :ts)"
        ), {"id": "t" * 32, "name": "default", "hash": "hash", "ts": "2026-01-01 00:00:00"})
        conn.execute(text(
            "INSERT INTO avatars (id, tenant_id, name, status, source_image_path, "
            "config_json, created_at) "
            "VALUES (:id, :tid, :name, :status, :path, :cfg, :ts)"
        ), {
            "id": "a" * 32, "tid": "t" * 32, "name": "alice", "status": "ready",
            "path": "/p", "cfg": "{}", "ts": "2026-01-01 00:00:00",
        })

    init_db(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        row = session.get(Avatar, "a" * 32)
        assert row is not None
        assert row.name == "alice"
        assert row.status == "ready"
    finally:
        session.close()
