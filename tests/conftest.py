import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DH_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DH_DRIVE_ENGINE", "mock")
    from server.config import get_settings
    get_settings.cache_clear()
    from server.db.session import init_db, get_engine, reset_engine
    reset_engine()
    from server.main import create_app
    app = create_app()
    init_db(get_engine())
    with TestClient(app) as c:
        yield c
    reset_engine()
    get_settings.cache_clear()
