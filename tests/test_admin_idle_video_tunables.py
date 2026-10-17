"""Idle-video admin route honours the tuned crf/preset.

The route rebuilds the idle clip on first read. It must pull crf/preset from
the same Setting table the admin UI writes; otherwise the operator-set quality
knob never reaches the encoder.
"""
from __future__ import annotations

import io
import json

from PIL import Image


def _upload_avatar(client, name):
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (200, 120, 80)).save(buf, format="PNG")
    return client.post(
        "/api/v1/avatars", data={"name": name},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]


def _make_video_mode(client, avatar_id):
    from pathlib import Path

    from server.config import get_settings

    root = Path(get_settings().data_dir) / "avatars" / avatar_id
    (root / "pack.json").write_text(
        json.dumps({"musetalk": {"mode": "video", "source": "source.mp4", "base_image": "normalized.png"}})
    )
    return root


def test_admin_idle_video_uses_tuned_preset(client, monkeypatch):
    from server.avatar.tunables import dump_json
    from server.db.models import Setting

    avatar_id = _upload_avatar(client, "a")
    _make_video_mode(client, avatar_id)
    # Operator pushed the quality knob.
    client.patch("/api/admin/settings", json={"idle_crf": 12, "idle_preset": "slow"})

    calls = []

    def fake_build(avatar_dir, source, settings, *, crf=None, preset=None):
        calls.append({"crf": crf, "preset": preset})
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("server.avatar.video.build_idle_video", fake_build)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 200
    assert calls, "build_idle_video was not called"
    assert calls[0]["crf"] == 12
    assert calls[0]["preset"] == "slow"


def test_admin_idle_video_honours_per_avatar_overrides(client, monkeypatch):
    # The create form stores its idle_crf / idle_preset choices on the avatar
    # row. A later re-encode of the idle clip must read those (not the
    # global Setting row), so the avatar's effective quality knob survives
    # the build.
    avatar_id = _upload_avatar(client, "a")
    _make_video_mode(client, avatar_id)
    # Push a global override that should be beaten by the per-avatar one.
    client.patch("/api/admin/settings", json={"idle_crf": 30})

    # Stamp per-avatar overrides directly (the create endpoint wires this).
    from sqlalchemy.orm import sessionmaker

    from server.config import get_settings
    from server.db.session import get_engine, reset_engine
    from server.db.models import Avatar

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        row = session.get(Avatar, avatar_id)
        row.tunable_overrides = {"idle_crf": 14, "idle_preset": "veryslow"}
        session.commit()
    finally:
        session.close()

    calls = []

    def fake_build(avatar_dir, source, settings, *, crf=None, preset=None):
        calls.append({"crf": crf, "preset": preset})
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("server.avatar.video.build_idle_video", fake_build)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 200
    assert calls, "build_idle_video was not called"
    # Per-avatar override wins over the Setting row.
    assert calls[0]["crf"] == 14
    assert calls[0]["preset"] == "veryslow"