"""MiniMax TTS config helper: layer DB rows on top of the built-in defaults.

The DB rows live in the ``settings`` table (one row per key) so they share
the persistence + admin-edit surface with the existing avatar tunables.
This module owns read/write + per-field validation; the synthesise path
calls ``read_config(db)`` once per request so the change takes effect
without a server restart.

There is no env layer any more: the Models tab's dialog is the only
entry point, exactly like the LLM.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def _fresh_db(tmp_path, monkeypatch):
    """Apply the standard test env (data_dir + DB in tmp) and give back an
    initialised SQLAlchemySession for the test to drive.
    """
    from tests.env import apply_env
    from server.config import get_settings
    from server.db.session import init_db, get_engine, reset_engine

    apply_env(monkeypatch, tmp_path)
    reset_engine()
    from server.main import create_app  # noqa: F401  -- ensures app init path
    init_db(get_engine())
    get_settings.cache_clear()
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=get_engine())
    db = Session()
    try:
        yield db
    finally:
        db.close()
        reset_engine()
        get_settings.cache_clear()


def test_read_config_returns_defaults_when_db_empty(_fresh_db):
    """No DB rows → the helper returns the built-in defaults.

    The api_key default is empty (a fresh install has no credential);
    model / sample_rate carry real defaults. ``voice_id`` used to live
    here -- the operator now picks it per session in the debug page, so
    the config dict no longer surfaces it.
    """
    from server.tts.config import read_config

    cfg = read_config(_fresh_db)
    assert cfg["api_key"] == ""
    assert "voice_id" not in cfg
    assert cfg["sample_rate"] == 32000


def test_read_config_layers_db_rows_over_defaults(_fresh_db):
    """A DB row for one key overrides the built-in default for that key.

    Other keys still come from the defaults so a partial update doesn't
    accidentally blank unrelated fields.
    """
    from server.tts.config import read_config
    from server.db.models import Setting

    _fresh_db.add(Setting(key="minimax_model", value=json.dumps("speech-02-turbo")))
    _fresh_db.commit()

    cfg = read_config(_fresh_db)
    assert cfg["model"] == "speech-02-turbo"
    # Other keys still come from the defaults.
    assert cfg["sample_rate"] == 32000


def test_write_config_persists_each_key(_fresh_db):
    """``write_config`` persists each non-empty field to the ``settings``
    table. A subsequent ``read_config`` returns the same values."""
    from server.tts.config import read_config, write_config
    from sqlalchemy import select
    from server.db.models import Setting

    cfg = write_config(
        _fresh_db,
        {
            "api_key": "test-key-1234",
            "model": "speech-02-turbo",
            "sample_rate": 24000,
            "base_url": "https://api.example.com",
            "group_id": "g-test",
        },
    )
    assert cfg["api_key"] == "test-key-1234"
    assert cfg["model"] == "speech-02-turbo"
    # All five rows landed in the settings table.
    rows = _fresh_db.execute(select(Setting).where(Setting.key.startswith("minimax_"))).scalars().all()
    assert {r.key for r in rows} >= {
        "minimax_api_key",
        "minimax_model",
        "minimax_sample_rate",
        "minimax_base_url",
        "minimax_group_id",
    }
    # voice_id must NOT be one of the persisted keys anymore.
    assert "minimax_voice_id" not in {r.key for r in rows}
    # And a fresh read sees them.
    cfg2 = read_config(_fresh_db)
    assert cfg2["model"] == "speech-02-turbo"
    assert cfg2["sample_rate"] == 24000


def test_write_config_rejects_invalid_sample_rate(_fresh_db):
    """``sample_rate`` must be one of the MiniMax-supported rates.

    Out-of-range values raise a clear ``AppError`` carrying a per-field
    message so the UI can show it inline.
    """
    from server.tts.config import write_config
    from server.core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        write_config(_fresh_db, {"sample_rate": 11025})  # not a MiniMax-supported rate
    err = excinfo.value
    # The per-field message lives on ``detail`` so the UI can show the
    # exact field + reason inline. Top-level message is the summary.
    assert err.detail is not None
    assert "sample_rate" in err.detail
    assert "8000" in err.detail["sample_rate"] or "supported" in err.detail["sample_rate"].lower()


def test_write_config_rejects_voice_id_field(_fresh_db):
    """``voice_id`` used to be a configurable field -- it isn't anymore.
    ``validate()`` rejects the unknown key with a per-field message so a
    stale client (still sending voice_id) gets a clear 400 rather than a
    silently dropped key.
    """
    from server.tts.config import validate

    errors = validate({"voice_id": "female-shaonv"})
    assert "voice_id" in errors
    assert errors["voice_id"] == "unknown field"