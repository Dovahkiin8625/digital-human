import io

import pytest
from PIL import Image

ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_admin_requires_token(client):
    assert client.get("/api/admin/system").status_code == 401


def test_tenant_and_key_roundtrip(client):
    resp = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "acme"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["api_key"].startswith("dh_")
    assert client.get("/api/v1/avatars", headers={"X-API-Key": body["api_key"]}).status_code == 200


def test_system_info(client):
    body = client.get("/api/admin/system", headers=ADMIN).json()
    assert body["engine"]["name"] == "mock"
    assert body["gpu"] is False


def test_avatar_config_and_preview(client):
    client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t"})
    api_key = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t2"}).json()["api_key"]
    avatar_id = client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": "a"},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/config", headers=ADMIN, json={"mouth_scale": 1.5}
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["mouth_scale"] == 1.5

    resp = client.get(f"/api/admin/avatars/{avatar_id}/preview.png?level=0.8", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


def test_jobs_listing_and_retry(client):
    api_key = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t"}).json()["api_key"]
    client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": "bad"},
        files={"file": ("bad.png", io.BytesIO(b"not an image"), "image/png")},
    )
    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    assert isinstance(jobs, list)
