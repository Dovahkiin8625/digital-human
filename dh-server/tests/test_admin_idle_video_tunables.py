"""Idle-video admin route honours the tuned crf/preset.

The route rebuilds the idle clip on first read. It must pull crf/preset from
the avatar's own tunable overrides -- what the create form captured -- or the
operator's quality choice never reaches the encoder.
"""
from __future__ import annotations

import io
import json

from PIL import Image


def _upload_avatar(client, name, **knobs):
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (200, 120, 80)).save(buf, format="PNG")
    data = {"name": name, "model": "musetalk"}
    data.update({k: str(v) for k, v in knobs.items()})
    return client.post(
        "/api/admin/avatars", data=data,
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    ).json()["avatar_id"]


def _make_video_mode(client, avatar_id):
    from pathlib import Path

    from dh_server.config import get_settings

    root = Path(get_settings().data_dir) / "avatars" / avatar_id
    (root / "pack.json").write_text(
        json.dumps({"musetalk": {"mode": "video", "source": "source.mp4", "base_image": "normalized.png"}})
    )
    return root


def test_admin_idle_video_uses_the_chosen_knobs(client, monkeypatch):
    # The create form captures the quality knob on the avatar row.
    avatar_id = _upload_avatar(client, "a", idle_crf=12, idle_preset="slow")
    _make_video_mode(client, avatar_id)

    calls = []

    def fake_build(avatar_dir, source, settings, *, crf=None, preset=None):
        calls.append({"crf": crf, "preset": preset})
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("dh_server.avatar.video.build_idle_video", fake_build)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 200
    assert calls, "build_idle_video was not called"
    assert calls[0]["crf"] == 12
    assert calls[0]["preset"] == "slow"


def test_admin_idle_video_honours_per_avatar_overrides(client, monkeypatch):
    # The create form stores its idle_crf / idle_preset choices on the avatar
    # row. A later re-encode of the idle clip must read those, so the
    # avatar's effective quality knob survives the build.
    avatar_id = _upload_avatar(client, "a")
    _make_video_mode(client, avatar_id)

    # Stamp per-avatar overrides directly (the create endpoint wires this).
    from sqlalchemy.orm import sessionmaker

    from dh_server.config import get_settings
    from dh_server.db.session import get_engine
    from dh_server.db.models import Avatar

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

    monkeypatch.setattr("dh_server.avatar.video.build_idle_video", fake_build)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 200
    assert calls, "build_idle_video was not called"
    assert calls[0]["crf"] == 14
    assert calls[0]["preset"] == "veryslow"