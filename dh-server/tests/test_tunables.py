"""Tunables schema: defaults, validation, and the per-avatar overrides.

These tiny tests pin down the contract the rest of the codebase relies on:
every default is finite, every range rejects values outside it, and any new
entry in ``TUNABLES`` is automatically exposed via ``defaults()`` /
``validate()`` without callers having to special-case it.

The tunables are per-avatar: the create form captures them and they are
stored on the avatar row, so there is no global layer to look up.
"""
from __future__ import annotations

from dh_server.avatar import tunables


def test_defaults_includes_idle_video_keys():
    # These two knobs live as tunables because every avatar build reaches
    # the idle encoder; the create form captures them per avatar.
    defaults = tunables.defaults()
    assert defaults["idle_crf"] == 18
    assert defaults["idle_preset"] == "veryfast"


def test_the_tunables_are_exactly_the_two_that_are_honoured():
    # Every entry in TUNABLES is read by something. A knob nothing reads
    # looks like a setting and is not one.
    assert {p.name for p in tunables.TUNABLES} == {"idle_crf", "idle_preset"}


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


def test_effective_applies_the_per_avatar_overrides():
    # The create form's idle_crf/idle_preset land on the avatar row and
    # beat the module defaults. Keys the form left out keep the default.
    out = tunables.effective(overrides={"idle_crf": 12, "idle_preset": "slow"})
    assert out["idle_crf"] == 12
    assert out["idle_preset"] == "slow"
    assert set(out) == {"idle_crf", "idle_preset"}


def test_effective_overrides_none_matches_the_default_call():
    # Passing overrides=None is identical to the no-overrides call; the
    # pipeline calls it that way.
    assert tunables.effective(overrides=None) == tunables.effective()


def test_init_db_adds_tunable_overrides_to_existing_avatars_table(tmp_path, monkeypatch):
    # Production deploys have an avatars table that predates the
    # tunable_overrides column. SQLAlchemy's ``create_all`` won't add the
    # column for them, so ``init_db`` must issue an idempotent ALTER TABLE.
    # We simulate the "old" schema by creating the table without the new
    # column, then call init_db and expect the column to be present + usable.
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from sqlalchemy import Column, MetaData, String, Table, create_engine, inspect

    from dh_server.config import get_settings
    from dh_server.db.session import init_db

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

    # The migration must add the new column to an already-present table,
    # and drop the columns the current model no longer carries.
    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "tunable_overrides" in cols
    assert "pack_path" not in cols

    # Round-trip an Avatar through the ORM to confirm the schema is usable.
    from dh_server.db.models import Avatar
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

    from dh_server.db.session import init_db

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    init_db(engine)
    init_db(engine)  # second call must not raise
