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
