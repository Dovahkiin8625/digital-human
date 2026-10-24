"""Avatar jobs audit the tunables they actually used.

Each completed job writes the tunables that applied (post-override, post-default)
into the avatar's config_json under ``_effective_params``. The UI reads this back
to show the operator exactly what their knobs translated into for this build.
"""
import json

import pytest
from PIL import Image

from doubles import FakePipeline
from server.avatar.pipeline import run_avatar_job
from server.avatar.store import AvatarStore
from server.db.models import Avatar, AvatarJob, Setting


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


def test_avatar_job_records_effective_params(tmp_path, db):
    # Operator-side overrides on two keys.
    from server.avatar.tunables import dump_json

    db.add(Setting(key="face_crop_size", value=dump_json(320)))
    db.add(Setting(key="idle_crf", value=dump_json(16)))
    db.commit()

    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)
    avatar = Avatar(name="a", status="processing", source_image_path=str(src))
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    run_avatar_job(db, store, FakePipeline(), job.id)
    db.refresh(avatar)

    params = avatar.config_json["_effective_params"]
    assert params["face_crop_size"] == 320
    assert params["idle_crf"] == 16
    # Untouched knob falls back to its module default.
    assert params["idle_preset"] == "veryfast"


def test_avatar_job_records_defaults_when_no_overrides(tmp_path, db):
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)
    avatar = Avatar(name="a", status="processing", source_image_path=str(src))
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    run_avatar_job(db, store, FakePipeline(), job.id)
    db.refresh(avatar)

    params = avatar.config_json["_effective_params"]
    assert params["face_crop_size"] == 256
    assert params["idle_crf"] == 18
    assert params["idle_preset"] == "veryfast"
    assert params["min_face_width_px"] == 160.0


def test_avatar_job_uses_per_avatar_overrides(tmp_path, db):
    # Per-avatar overrides (from the create form) beat both defaults and any
    # operator-pushed Setting rows. The audit row reflects what the pipeline
    # actually saw, so a re-encode of the idle clip later reads the same
    # numbers from avatar.tunable_overrides.
    from server.avatar.tunables import dump_json

    db.add(Setting(key="idle_crf", value=dump_json(22)))
    db.commit()

    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)
    avatar = Avatar(
        name="a",
        status="processing",
        source_image_path=str(src),
        tunable_overrides={"idle_crf": 12, "idle_preset": "slow"},
    )
    db.add(avatar)
    db.flush()
    job = AvatarJob(avatar_id=avatar.id, stage="queued")
    db.add(job)
    db.commit()

    run_avatar_job(db, store, FakePipeline(), job.id)
    db.refresh(avatar)

    params = avatar.config_json["_effective_params"]
    assert params["idle_crf"] == 12
    assert params["idle_preset"] == "slow"
    # The DB-stored override was beaten by the per-avatar one.
    assert params["idle_crf"] != 22


def test_avatar_tunable_overrides_default_is_empty(tmp_path, db):
    # New avatars must come back with an empty dict, not NULL, so the pipeline
    # can blindly pass ``avatar.tunable_overrides`` to ``effective()`` without
    # a None-check.
    store = AvatarStore(tmp_path / "data")
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (200, 120, 80)).save(src)
    avatar = Avatar(name="a", source_image_path=str(src))
    db.add(avatar)
    db.commit()
    db.refresh(avatar)
    assert avatar.tunable_overrides == {}
