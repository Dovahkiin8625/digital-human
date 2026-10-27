"""GenerationTask: the unified task row for async, long-running API operations.

Currently only the i2v (图生视频) endpoint surfaces a GenerationTask. The row
shape is deliberately generic (type, status, stage, progress, error, params,
result) so a future long op just declares its own ``type`` and reuses the
table -- no new schema needed.
"""
from __future__ import annotations

import json

import pytest


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


def test_generation_task_round_trips_status_and_progress(db):
    from server.db.models import GenerationTask

    task = GenerationTask(
        id="t" * 32,
        type="i2v",
        status="running",
        stage="encoding",
        progress=42,
        params_json=json.dumps({"backend": "liveportrait", "prompt": "hi"}),
    )
    db.add(task)
    db.commit()

    db.refresh(task)
    assert task.type == "i2v"
    assert task.status == "running"
    assert task.stage == "encoding"
    assert task.progress == 42
    # params_json is stored as a JSON string and deserialised back as a dict.
    assert json.loads(task.params_json) == {"backend": "liveportrait", "prompt": "hi"}
    # Result column starts NULL and is filled in once the animator writes the
    # mp4; the bytes then travel back through ``GET /api/admin/tasks/{id}/result``.
    assert task.result_json is None
    assert task.error is None


def test_generation_task_default_status_is_queued(db):
    # New tasks default to ``queued`` so the API caller can immediately
    # differentiate ``done`` from "not even started yet" without a NULL check.
    from server.db.models import GenerationTask

    task = GenerationTask(id="q" * 32, type="i2v", params_json="{}")
    db.add(task)
    db.commit()
    db.refresh(task)
    assert task.status == "queued"
    assert task.progress == 0
    assert task.stage == ""


def test_generation_task_records_failure(db):
    # Failed tasks carry the error message and keep their params around so an
    # operator can replay them from the Tasks tab.
    from server.db.models import GenerationTask

    task = GenerationTask(
        id="f" * 32,
        type="i2v",
        status="failed",
        error="weights missing",
        params_json=json.dumps({"backend": "liveportrait"}),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    assert task.status == "failed"
    assert task.error == "weights missing"