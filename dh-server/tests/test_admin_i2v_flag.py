"""``animate`` upload flag: opt-in to image-to-video before MuseTalk prep.

The flag is carried end-to-end from the upload form into the avatar row so the
pipeline can pick it up after the job is re-queued, e.g. on retry. For video
uploads the flag is silently ignored: the upload modal never shows the toggle
for a ``.mp4``, but we cannot trust the client.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from dh_server.config import get_settings


ADMIN = {"X-Admin-Token": "dev-admin-token"}


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _tiny_mp4(tmp_path: Path) -> bytes:
    """Build a 1s, 25 fps, 720x1280 mp4 the store's validator accepts.

    Uses the existing PyAV libx264 path on a constant-colour frame so the test
    stays GPU-free and fast.
    """
    import numpy as np
    from dh_server.avatar.video import _encode_h264

    out = tmp_path / "tiny.mp4"
    frames = np.full((25, 1280, 720, 3), 30, dtype=np.uint8)
    _encode_h264(out, frames, fps=25, crf=28, preset="ultrafast")
    return out.read_bytes()


def _avatar_row(client, avatar_id: str) -> dict:
    return client.get(f"/api/admin/avatars/{avatar_id}", headers=ADMIN).json()


def test_admin_animate_default_is_false(client):
    # Without ``animate`` in the form data, the avatar row must not opt in.
    resp = client.post(
        "/api/admin/avatars",
        headers=ADMIN,
        data={"name": "img"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 201
    detail = _avatar_row(client, resp.json()["avatar_id"])
    assert detail["config"].get("request", {}).get("animate") is False


def test_admin_animate_true_persists_in_config_json(client):
    resp = client.post(
        "/api/admin/avatars",
        headers=ADMIN,
        data={"name": "img", "animate": "true"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 201
    detail = _avatar_row(client, resp.json()["avatar_id"])
    assert detail["config"]["request"]["animate"] is True


def test_admin_video_upload_ignores_animate_flag(client, tmp_path):
    # The form field is hidden in the UI for a video, but a curl request
    # must not flip a video avatar into the i2v branch (no source frame to
    # diffuse from).
    resp = client.post(
        "/api/admin/avatars",
        headers=ADMIN,
        data={"name": "v", "animate": "true"},
        files={"file": ("v.mp4", _tiny_mp4(tmp_path), "video/mp4")},
    )
    assert resp.status_code == 201
    detail = _avatar_row(client, resp.json()["avatar_id"])
    assert detail["config"].get("request", {}).get("animate") is False


def test_tenant_animate_true_persists_in_config_json(client):
    api_key = client.post(
        "/api/admin/tenants", headers=ADMIN, json={"name": "t"}
    ).json()["api_key"]
    resp = client.post(
        "/api/v1/avatars",
        headers={"X-API-Key": api_key},
        data={"name": "img", "animate": "true"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    assert resp.status_code == 201
    # The tenant surface doesn't surface request metadata; verify directly.
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from dh_server.db.models import Avatar
    from dh_server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        avatar = db.execute(select(Avatar).where(Avatar.id == resp.json()["avatar_id"])).scalar_one()
        assert avatar.config_json.get("request", {}).get("animate") is True
    finally:
        db.close()


def test_avatar_retry_preserves_animate_flag(client):
    # Retry must not silently drop a previously-true animate flag, otherwise
    # an operator rebuilding an avatar loses the i2v path on the second go.
    resp = client.post(
        "/api/admin/avatars",
        headers=ADMIN,
        data={"name": "img", "animate": "true"},
        files={"file": ("a.png", _png(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]

    detail_before = _avatar_row(client, avatar_id)
    assert detail_before["config"]["request"]["animate"] is True

    client.post(f"/api/admin/avatars/{avatar_id}/retry", headers=ADMIN)
    detail_after = _avatar_row(client, avatar_id)
    assert detail_after["config"]["request"]["animate"] is True
