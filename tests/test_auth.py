from server.api.deps import hash_api_key
from server.db.models import Tenant


def test_error_envelope_shape(client):
    resp = client.get("/api/v1/avatars")
    assert resp.status_code == 401
    body = resp.json()
    assert body == {
        "error": {"code": "unauthorized", "message": "missing or invalid API key", "detail": None}
    }


def test_admin_login_wrong_password(client):
    resp = client.post("/api/admin/login", json={"password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_admin_login_ok(client):
    resp = client.post("/api/admin/login", json={"password": "admin123"})
    assert resp.status_code == 200
    assert resp.json()["token"] == "dev-admin-token"


def test_hash_api_key_stable():
    assert hash_api_key("k1") == hash_api_key("k1")
    assert hash_api_key("k1") != hash_api_key("k2")


def test_tenant_auth_ok(client):
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key("key-1")))
    db.commit()
    db.close()
    resp = client.get("/api/v1/avatars", headers={"X-API-Key": "key-1"})
    assert resp.status_code == 200


def test_admin_login_wrong_password_non_ascii(client):
    resp = client.post("/api/admin/login", json={"password": "ü"})
    assert resp.status_code == 401
    body = resp.json()
    assert body == {
        "error": {"code": "unauthorized", "message": "missing or invalid admin token", "detail": None}
    }


def test_get_admin_non_ascii_token_401_envelope():
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from server.api.deps import get_admin
    from server.core.errors import register_handlers

    app = FastAPI()
    register_handlers(app)

    @app.get("/admin/thing", dependencies=[Depends(get_admin)])
    def admin_thing() -> dict:
        return {"ok": True}

    # raw high-bit header bytes: TestClient refuses non-ASCII str header values
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/admin/thing", headers={"X-Admin-Token": "ü".encode("latin-1")})
    assert resp.status_code == 401
    body = resp.json()
    assert body == {
        "error": {"code": "unauthorized", "message": "missing or invalid admin token", "detail": None}
    }


def test_internal_error_envelope_hides_exception_text():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.core.errors import register_handlers

    app = FastAPI()
    register_handlers(app)

    @app.get("/boom")
    def boom():
        raise RuntimeError("s3cret-boom")

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error": {"code": "internal_error", "message": "internal server error", "detail": None}
    }
    assert "s3cret-boom" not in resp.text
