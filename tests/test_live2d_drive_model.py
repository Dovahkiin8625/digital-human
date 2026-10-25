"""`Avatar.drive_model`: the column, its migration, and its filter.

The column is the pairing between an avatar and the digital-human model
that drives it. It replaced ``Avatar.kind``, whose two values
("photoreal" / "live2d") were really two drive models wearing a
different name.

The rebuild in ``_drop_abandoned_columns`` recreates ``avatars`` from a
whitelist of column names. A column missing from that whitelist is
silently dropped the first time an old database triggers the rebuild --
no error, no log, the column just disappears.
"""

import io

from PIL import Image
from sqlalchemy import inspect

from server.db import session as session_mod
from server.db.session import get_engine, init_db


def _png_bytes(w=640, h=800):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_drive_model_defaults_to_musetalk(client):
    resp = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    assert client.get(f"/api/v1/avatars/{avatar_id}").json()["drive_model"] == "musetalk"


def test_migration_is_idempotent():
    engine = get_engine()
    init_db(engine)
    init_db(engine)  # second run must be a no-op, not an error
    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "drive_model" in cols


def test_tenant_rebuild_whitelist_includes_drive_model():
    """Pin the whitelist itself. Adding a column without updating it fails
    here rather than in production."""
    import inspect as pyinspect

    source = pyinspect.getsource(session_mod._drop_abandoned_columns)
    assert '"drive_model"' in source, "avatars rebuild whitelist is missing 'drive_model'"


def test_drive_model_filter_partitions_the_listing(client):
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(name="l2d", status="ready", source_image_path="", drive_model="live2d"))
        db.add(Avatar(name="photo", status="ready", source_image_path="", drive_model="musetalk"))
        db.commit()
    finally:
        db.close()

    everything = client.get("/api/v1/avatars").json()
    live2d = client.get("/api/v1/avatars", params={"drive_model": "live2d"}).json()
    photoreal = client.get("/api/v1/avatars", params={"drive_model": "musetalk"}).json()

    assert len(everything) == 2
    assert [a["name"] for a in live2d] == ["l2d"]
    assert [a["name"] for a in photoreal] == ["photo"]


def test_the_kind_column_is_gone_and_its_values_moved(client):
    """An old database carries ``kind``; the migration moves the value into
    ``drive_model`` and drops the column.

    The migration runs on ``init_db``, so this test drives it directly
    rather than through the ORM: the ORM has no ``kind`` attribute any
    more, so a row cannot be seeded the old way.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE avatars ADD COLUMN kind VARCHAR(16) NOT NULL DEFAULT 'photoreal'"
        )
        conn.exec_driver_sql(
            "INSERT INTO avatars (id, name, status, source_image_path, pack_path, "
            "config_json, tunable_overrides, drive_model, created_at, kind) "
            "VALUES ('b' || substr(hex(randomblob(16)), 1, 31), 'old', 'ready', '', "
            "NULL, '{}', '{}', 'musetalk', '2026-01-01 00:00:00', 'live2d')"
        )

    init_db(engine)

    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "kind" not in cols
    assert "drive_model" in cols
    with engine.begin() as conn:
        row = conn.exec_driver_sql(
            "SELECT name, drive_model FROM avatars WHERE name = 'old'"
        ).fetchone()
    assert row is not None
    assert row[1] == "live2d"
