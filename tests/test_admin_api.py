import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _png(size=(640, 800)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _new_tenant(client, name):
    return client.post("/api/admin/tenants", headers=ADMIN, json={"name": name}).json()["api_key"]


def _upload_avatar(client, api_key, name):
    return client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": name},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]


def _admin_avatar(client, name="a", tenant_id="", size=(640, 800)):
    data = {"name": name}
    if tenant_id:
        data["tenant_id"] = tenant_id
    return client.post(
        "/api/admin/avatars", headers=ADMIN, data=data,
        files={"file": ("a.png", _png(size), "image/png")},
    )


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


def test_admin_issue_api_key_returns_usable_key(client):
    tenant_id = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "acme"}).json()[
        "tenant_id"
    ]

    resp = client.post(f"/api/admin/tenants/{tenant_id}/api-key", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == tenant_id
    assert body["api_key"].startswith("dh_")
    # The freshly issued key must authenticate against the real customer surface.
    assert (
        client.get("/api/v1/avatars", headers={"X-API-Key": body["api_key"]}).status_code == 200
    )


def test_admin_issue_api_key_rotates_previous(client):
    created = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "acme"}).json()
    tenant_id, first_key = created["tenant_id"], created["api_key"]
    assert client.get("/api/v1/avatars", headers={"X-API-Key": first_key}).status_code == 200

    second_key = client.post(
        f"/api/admin/tenants/{tenant_id}/api-key", headers=ADMIN
    ).json()["api_key"]

    assert second_key != first_key
    # Rotation invalidates the previous key but keeps the new one working.
    assert client.get("/api/v1/avatars", headers={"X-API-Key": first_key}).status_code == 401
    assert client.get("/api/v1/avatars", headers={"X-API-Key": second_key}).status_code == 200


def test_admin_issue_api_key_unknown_tenant(client):
    resp = client.post(f"/api/admin/tenants/{uuid4().hex}/api-key", headers=ADMIN)
    assert resp.status_code == 404


def test_system_info(client):
    body = client.get("/api/admin/system", headers=ADMIN).json()
    assert body["engine"]["name"] == "mock"
    assert body["pipeline"] == "mock"
    assert body["gpu"] is False


def test_avatar_config(client):
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


def _make_video_mode(client, avatar_id):
    """Rewrite the uploaded avatar's pack.json into a video-mode pack."""
    from server.config import get_settings

    root = Path(get_settings().data_dir) / "avatars" / avatar_id
    (root / "pack.json").write_text(
        json.dumps(
            {"musetalk": {"mode": "video", "source": "source.mp4", "base_image": "normalized.png"}}
        )
    )
    return root


def test_normalized_png_is_served(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/normalized.png", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 50


def test_idle_mp4_is_generated_then_served_with_range(client, monkeypatch):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    root = _make_video_mode(client, avatar_id)

    def fake_build(avatar_dir, source, settings):
        assert source == avatar_dir / "source.mp4"
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("server.avatar.video.build_idle_video", fake_build)

    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.content == b"FAKEMP4-DATA"

    # A ranged read is served by FileResponse (206 Partial Content).
    rng = client.get(
        f"/api/admin/avatars/{avatar_id}/idle.mp4", headers={**ADMIN, "Range": "bytes=0-3"}
    )
    assert rng.status_code == 206
    assert rng.content == b"FAKE"


def test_idle_mp4_build_failure_is_503(client, monkeypatch):
    from server.avatar.video import VideoError

    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    _make_video_mode(client, avatar_id)

    def boom(avatar_dir, source, settings):
        raise VideoError("decoder died")

    monkeypatch.setattr("server.avatar.video.build_idle_video", boom)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "idle_video_unavailable"


def test_idle_mp4_404_for_photo_avatar(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 404


def test_preview_png_route_is_removed(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/preview.png", headers=ADMIN)
    assert resp.status_code == 404


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


def test_admin_create_avatar_pending_and_schedules_job(client, monkeypatch):
    # TestClient runs BackgroundTasks synchronously, so a real run would flip the
    # avatar to ready before we could observe the creation state. Stub the runner
    # to assert both the pending row and that the job was actually scheduled.
    scheduled = []
    monkeypatch.setattr(
        "server.api.admin.run_job_in_new_session",
        lambda job_id, store: scheduled.append(job_id),
    )

    resp = _admin_avatar(client, name="hero")
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    avatar_id = body["avatar_id"]

    rows = client.get("/api/admin/avatars", headers=ADMIN).json()
    row = next(r for r in rows if r["avatar_id"] == avatar_id)
    assert row["status"] == "pending"

    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    job = next(j for j in jobs if j["avatar_id"] == avatar_id)
    assert scheduled == [job["job_id"]]


def test_admin_create_avatar_explicit_tenant(client):
    tenant_id = client.post(
        "/api/admin/tenants", headers=ADMIN, json={"name": "acme"}
    ).json()["tenant_id"]

    resp = _admin_avatar(client, name="a", tenant_id=tenant_id)
    assert resp.status_code == 201
    detail = client.get(
        f"/api/admin/avatars/{resp.json()['avatar_id']}", headers=ADMIN
    ).json()
    assert detail["tenant_id"] == tenant_id


def test_admin_create_avatar_creates_default_tenant(client):
    assert client.get("/api/admin/tenants", headers=ADMIN).json() == []

    resp = _admin_avatar(client, name="a")
    assert resp.status_code == 201

    defaults = [
        t
        for t in client.get("/api/admin/tenants", headers=ADMIN).json()
        if t["name"] == "default"
    ]
    assert len(defaults) == 1

    detail = client.get(
        f"/api/admin/avatars/{resp.json()['avatar_id']}", headers=ADMIN
    ).json()
    assert detail["tenant_id"] == defaults[0]["tenant_id"]


def test_admin_create_avatar_reuses_default_tenant(client):
    first = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "default"}).json()

    resp = _admin_avatar(client, name="a")
    assert resp.status_code == 201

    defaults = [
        t
        for t in client.get("/api/admin/tenants", headers=ADMIN).json()
        if t["name"] == "default"
    ]
    assert len(defaults) == 1
    assert defaults[0]["tenant_id"] == first["tenant_id"]

    detail = client.get(
        f"/api/admin/avatars/{resp.json()['avatar_id']}", headers=ADMIN
    ).json()
    assert detail["tenant_id"] == first["tenant_id"]


def test_admin_get_avatar_detail_and_error(client):
    api_key = _new_tenant(client, "t")
    good = _upload_avatar(client, api_key, "good")
    detail = client.get(f"/api/admin/avatars/{good}", headers=ADMIN).json()
    assert set(detail) == {
        "avatar_id",
        "tenant_id",
        "name",
        "status",
        "config",
        "error",
        "created_at",
    }
    assert detail["avatar_id"] == good
    assert detail["config"]["mouth_scale"] == 1.0
    assert detail["error"] is None

    bad = client.post(
        "/api/v1/avatars",
        headers={"X-API-Key": api_key},
        data={"name": "bad"},
        files={"file": ("b.png", _png((50, 50)), "image/png")},
    ).json()["avatar_id"]
    failed = client.get(f"/api/admin/avatars/{bad}", headers=ADMIN).json()
    assert failed["status"] == "failed"
    assert failed["error"]


def test_admin_get_avatar_not_found(client):
    assert client.get(f"/api/admin/avatars/{uuid4().hex}", headers=ADMIN).status_code == 404
    assert client.get("/api/admin/avatars/not-a-uuid", headers=ADMIN).status_code == 404


def test_admin_delete_avatar(client):
    from server.avatar.store import AvatarStore
    from server.config import get_settings

    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    store = AvatarStore(get_settings().data_dir)
    assert store.avatar_dir(avatar_id).exists()

    resp = client.delete(f"/api/admin/avatars/{avatar_id}", headers=ADMIN)
    assert resp.status_code == 204

    rows = client.get("/api/admin/avatars", headers=ADMIN).json()
    assert all(r["avatar_id"] != avatar_id for r in rows)
    jobs = client.get("/api/admin/jobs", headers=ADMIN).json()
    assert all(j["avatar_id"] != avatar_id for j in jobs)
    assert not store.avatar_dir(avatar_id).exists()


def test_admin_delete_avatar_not_found(client):
    assert client.delete(f"/api/admin/avatars/{uuid4().hex}", headers=ADMIN).status_code == 404
    assert client.delete("/api/admin/avatars/not-a-uuid", headers=ADMIN).status_code == 404
