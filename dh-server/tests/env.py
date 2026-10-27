"""Shared test environment setup.

The server reads its configuration from the environment; the tests point it
at a throwaway data dir and DB.

The digital-human model is *not* configured here. It is a registry entry
(``server/model_registry.py``), and the suite swaps the whole model for the
doubles in ``tests/doubles.py`` -- see ``tests/conftest.py``.
"""
from __future__ import annotations


def apply_env(monkeypatch, tmp_path, **extra):
    """Point settings at a throwaway data dir/DB."""
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    for key, value in extra.items():
        monkeypatch.setenv(key, str(value))
    from server.config import get_settings

    get_settings.cache_clear()
