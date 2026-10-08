import io

import pytest
from PIL import Image

ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _new_tenant(client, name):
    return client.post("/api/admin/tenants", headers=ADMIN, json={"name": name}).json()["api_key"]


def _upload_avatar(client, api_key, name):
    return client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": name},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]


def _set_status(avatar_id, status):
    """Force an avatar's status so retry can be exercised from every allowed state."""
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one()
        avatar.status = status
        db.commit()
    finally:
        db.close()


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


def test_jobs_listing(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    assert isinstance(jobs, list)
    row = next(j for j in jobs if j["avatar_id"] == avatar_id)
    assert set(row) >= {"job_id", "avatar_id", "stage", "progress", "error", "created_at"}


@pytest.mark.parametrize("status", ["failed", "pending", "ready"])
def test_avatar_retry_allowed_from_any_status(client, status):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    _set_status(avatar_id, status)

    resp = client.post(f"/api/admin/avatars/{avatar_id}/retry", headers=ADMIN)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # A brand-new job row exists for this avatar (the retry background task ran).
    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    assert any(j["job_id"] == job_id and j["avatar_id"] == avatar_id for j in jobs)
