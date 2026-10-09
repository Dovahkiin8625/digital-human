"""Shared test environment setup.

Both the drive engine and the avatar pipeline default to their *real*
implementations in production (`musetalk`), and both must be pinned to their
test doubles here. Pinning only one silently makes every upload-path test fail
(the avatar never reaches `ready`, so session creation 409s), which is why
this lives in one place instead of being repeated per test module.
"""
from __future__ import annotations

TEST_DOUBLES = {
    "DH_DRIVE_ENGINE": "mock",
    "DH_AVATAR_PIPELINE": "mock",
}


def apply_env(monkeypatch, tmp_path, **extra):
    """Point settings at a throwaway data dir/DB and pin both test doubles."""
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    for key, value in {**TEST_DOUBLES, **extra}.items():
        monkeypatch.setenv(key, str(value))
    from server.config import get_settings

    get_settings.cache_clear()
