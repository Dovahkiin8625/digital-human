import pytest
from doubles import install_doubles
from env import apply_env
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    # The server ships exactly one digital-human model (MuseTalk), which
    # needs a GPU, torch and the weights. The suite swaps the whole model
    # for the doubles so it can run in the torch-free .venv.
    install_doubles(monkeypatch)
    from server.config import get_settings
    from server.db.session import init_db, get_engine, reset_engine
    reset_engine()
    from server.main import create_app
    app = create_app()
    init_db(get_engine())
    with TestClient(app) as c:
        yield c
    reset_engine()
    get_settings.cache_clear()
