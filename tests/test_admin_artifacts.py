"""Admin artifact/timeline listing endpoints.

The admin console surfaces on-disk artifacts (source, normalized, latents,
masks, idle clip, pack.json) plus the per-stage pipeline meta. Backend must
ship tiny list endpoints that walk the avatar's data dir + read musetalk/meta.json.
"""
from __future__ import annotations

import io

from PIL import Image


def _upload_avatar(client, name):
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (200, 120, 80)).save(buf, format="PNG")
    return client.post(
        "/api/v1/avatars", data={"name": name},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]


def test_artifacts_lists_pack_files_with_sizes(client):
    avatar_id = _upload_avatar(client, "a")

    resp = client.get(f"/api/admin/avatars/{avatar_id}/artifacts")
    assert resp.status_code == 200
    rows = resp.json()
    paths = {r["path"]: r for r in rows}
    # Mock pipeline writes normalized.png + pack.json; source.<suffix> too.
    assert "normalized.png" in paths
    assert "pack.json" in paths
    for r in rows:
        assert r["size"] >= 0
        assert r["kind"] in {"image", "json", "tensor", "mask", "video", "binary"}


def test_artifacts_rejects_unknown_avatar(client):
    from uuid import uuid4

    resp = client.get(f"/api/admin/avatars/{uuid4().hex}/artifacts")
    assert resp.status_code == 404
    resp = client.get("/api/admin/avatars/not-a-uuid/artifacts")
    assert resp.status_code == 404


def test_timeline_returns_pipeline_meta(client):
    avatar_id = _upload_avatar(client, "a")

    resp = client.get(f"/api/admin/avatars/{avatar_id}/timeline")
    assert resp.status_code == 200
    body = resp.json()
    # Either the mock pipeline (no musetalk dir) returns minimal meta, or the real
    # one returns the full musetalk/meta.json contents. Both shapes include mode.
    assert "mode" in body
    assert body["mode"] in {"image", "video"}
    # fps / version / detector are always either present in meta.json or unset.
    assert "fps" in body
    assert "crop" in body


def test_timeline_rejects_unknown_avatar(client):
    from uuid import uuid4

    resp = client.get(f"/api/admin/avatars/{uuid4().hex}/timeline")
    assert resp.status_code == 404


def test_timeline_includes_quality_warnings_when_present(client):
    avatar_id = _upload_avatar(client, "a")
    # Forge warnings into the avatar's config_json (mock pipeline path).
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        a = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one()
        a.config_json = {"crop": {"mode": "center"}, "warnings": ["脸太小"]}
        db.commit()
    finally:
        db.close()

    body = client.get(
        f"/api/admin/avatars/{avatar_id}/timeline",
    ).json()
    assert body.get("warnings") == ["脸太小"]