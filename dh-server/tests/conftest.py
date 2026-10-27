import os

import pytest
from doubles import install_doubles
from env import apply_env
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True, scope="session")
def _never_touch_the_real_database(tmp_path_factory):
    """Point the settings at a throwaway dir for the WHOLE session.

    The default ``database_url`` is relative to the working directory, so a
    test that opens the engine without the ``client`` fixture would
    otherwise create -- or open -- the database that sits in the service's
    real ``data/`` directory. Pointing the environment at a session-wide tmp
    dir makes that impossible; the per-test ``apply_env`` still overrides
    it, and monkeypatch restores this value afterwards.
    """
    root = tmp_path_factory.mktemp("session")
    os.environ["DH_DATA_DIR"] = str(root / "data")
    os.environ["DH_DATABASE_URL"] = f"sqlite:///{root / 'test.db'}"
    from dh_server.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    # The server-driven digital-human models (MuseTalk, the Wav2Lip CPU
    # engine) need torch and the weights. The suite swaps them for the
    # doubles so it can run in the torch-free .venv.
    install_doubles(monkeypatch)
    from dh_server.config import get_settings
    from dh_server.db.session import init_db, get_engine, reset_engine
    reset_engine()
    from dh_server.main import create_app
    app = create_app()
    init_db(get_engine())
    with TestClient(app) as c:
        yield c
    reset_engine()
    get_settings.cache_clear()
