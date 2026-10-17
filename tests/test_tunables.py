"""Tunables schema: defaults, validation, DB-backed ``effective()`` lookup.

These tiny tests pin down the contract the rest of the codebase relies on:
every default is finite, every range rejects values outside it, and any new
entry in ``TUNABLES`` is automatically exposed via ``defaults()`` /
``validate()`` without callers having to special-case it.
"""
from __future__ import annotations

import pytest

from server.avatar import tunables


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from sqlalchemy.orm import sessionmaker

    from server.config import get_settings
    from server.db.session import get_engine, init_db, reset_engine

    get_settings.cache_clear()
    reset_engine()
    init_db(get_engine())
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
    finally:
        session.close()
        reset_engine()
        get_settings.cache_clear()


def test_defaults_includes_idle_video_keys():
    # These two knobs still live as global tunables because every avatar
    # build reaches the idle encoder; the operator tunes them once and the
    # pipeline picks them up.
    defaults = tunables.defaults()
    assert defaults["idle_crf"] == 18
    assert defaults["idle_preset"] == "veryfast"


def test_idle_preset_enum_is_enforced():
    # Anything other than the known x264 presets is rejected; the encoder
    # silently swaps to a default if PyAV gets an unknown one, which would
    # 502 silently without this check.
    assert "idle_preset" in tunables.validate({"idle_preset": "not-a-preset"})
    assert tunables.validate({"idle_preset": "veryfast"}) == {}


def test_idle_crf_bounds_enforced():
    # libx264's documented range is [0, 51]; outside it the encoder issues a
    # warning but does not error, so the validator guards the contract.
    assert "idle_crf" in tunables.validate({"idle_crf": -1})
    assert "idle_crf" in tunables.validate({"idle_crf": 100})


def test_settings_endpoint_round_trips_idle_keys(client):
    # PATCH then GET must reflect the stored overrides; the response shape
    # matches GET's flat dict so the admin form can rehydrate cleanly.
    admin = {"X-Admin-Token": "dev-admin-token"}
    body = client.patch(
        "/api/admin/settings",
        headers=admin,
        json={"idle_crf": 22, "idle_preset": "fast"},
    ).json()
    assert body["idle_crf"] == 22
    assert body["idle_preset"] == "fast"

    fetched = client.get("/api/admin/settings", headers=admin).json()
    assert fetched["idle_crf"] == 22
    assert fetched["idle_preset"] == "fast"


def test_settings_endpoint_rejects_out_of_range_value(client):
    admin = {"X-Admin-Token": "dev-admin-token"}
    resp = client.patch("/api/admin/settings", headers=admin, json={"idle_crf": 999})
    assert resp.status_code == 400


def test_effective_accepts_overrides_that_beat_db_rows(db):
    # Per-avatar overrides (used by the create form's idle_crf/idle_preset
    # fields) must win over both the module defaults and any operator-pushed
    # Setting rows. Missing keys in the override dict keep falling through.
    from server.avatar.tunables import dump_json
    from server.db.models import Setting

    db.add(Setting(key="idle_crf", value=dump_json(22)))
    db.commit()

    out = tunables.effective(db, overrides={"idle_crf": 12, "idle_preset": "slow"})
    # Override wins over the DB row.
    assert out["idle_crf"] == 12
    # Override applies even when no DB row exists for that key.
    assert out["idle_preset"] == "slow"
    # Untouched keys keep the default.
    assert out["bbox_shift"] == 0


def test_effective_overrides_none_matches_legacy(db):
    # Passing overrides=None is identical to the old single-arg call; this is
    # the contract the pipeline (which doesn't yet know about overrides) relies
    # on for the duration of the migration.
    assert tunables.effective(db, overrides=None) == tunables.effective(db)


def test_init_db_adds_tunable_overrides_to_existing_avatars_table(tmp_path, monkeypatch):
    # Production deploys have an avatars table that predates the
    # tunable_overrides column. SQLAlchemy's ``create_all`` won't add the
    # column for them, so ``init_db`` must issue an idempotent ALTER TABLE.
    # We simulate the "old" schema by creating the table without the new
    # column, then call init_db and expect the column to be present + usable.
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from sqlalchemy import Column, MetaData, String, Table, create_engine

    from server.config import get_settings
    from server.db.session import init_db

    get_settings.cache_clear()
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    # Build the *old* shape (no tunable_overrides column) by hand.
    old_meta = MetaData()
    Table(
        "avatars",
        old_meta,
        Column("id", String(32), primary_key=True),
        Column("tenant_id", String(32)),
        Column("name", String(128)),
        Column("status", String(16)),
        Column("source_image_path", String(512)),
        Column("pack_path", String(512), nullable=True),
        Column("config_json", String(64)),
        Column("created_at", String(32)),
    )
    old_meta.create_all(engine)

    init_db(engine)

    # The migration must add the new column to an already-present table.
    # Round-trip an Avatar through the ORM to confirm the schema is usable.
    from server.db.models import Avatar
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        row = Avatar(id="x" * 32, name="a", source_image_path="/p", status="pending")
        session.add(row)
        session.commit()
        session.refresh(row)
        assert row.tunable_overrides == {}
        # Writing a real override round-trips through JSON.
        row.tunable_overrides = {"idle_crf": 12}
        session.commit()
        session.refresh(row)
        assert row.tunable_overrides == {"idle_crf": 12}
    finally:
        session.close()


def test_init_db_is_idempotent(tmp_path):
    # Calling init_db twice (or after a fresh create_all) must not error.
    # The migration helper must swallow the "duplicate column" case.
    from sqlalchemy import create_engine

    from server.db.session import init_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    init_db(engine)
    init_db(engine)  # second call must not raise
