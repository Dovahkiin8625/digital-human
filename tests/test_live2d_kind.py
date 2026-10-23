"""`Avatar.kind` migration: additive, idempotent, and safe against the
tenant-schema rebuild.

The rebuild in ``_drop_abandoned_tenant_schema`` recreates ``avatars`` from
a whitelist of column names. A column missing from that whitelist is
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


def test_kind_defaults_to_photoreal(client):
    resp = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    assert client.get(f"/api/v1/avatars/{avatar_id}").json()["kind"] == "photoreal"


def test_migration_is_idempotent():
    engine = get_engine()
    init_db(engine)
    init_db(engine)  # second run must be a no-op, not an error
    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "kind" in cols


def test_tenant_rebuild_whitelist_includes_kind():
    """Pin the whitelist itself. Adding a column without updating it fails
    here rather than in production."""
    import inspect as pyinspect

    source = pyinspect.getsource(session_mod._drop_abandoned_tenant_schema)
    assert '"kind"' in source, "avatars rebuild whitelist is missing 'kind'"


def test_kind_filter_partitions_the_listing(client):
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(name="l2d", status="ready", source_image_path="", kind="live2d"))
        db.add(Avatar(name="photo", status="ready", source_image_path="", kind="photoreal"))
        db.commit()
    finally:
        db.close()

    everything = client.get("/api/v1/avatars").json()
    live2d = client.get("/api/v1/avatars", params={"kind": "live2d"}).json()
    photoreal = client.get("/api/v1/avatars", params={"kind": "photoreal"}).json()

    assert len(everything) == 2
    assert [a["name"] for a in live2d] == ["l2d"]
    assert [a["name"] for a in photoreal] == ["photo"]