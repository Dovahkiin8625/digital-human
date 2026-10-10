import io

import pytest
from PIL import Image


def _png_bytes(w=640, h=800):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _seed_tenant(client, api_key="key-1"):
    from server.api.deps import hash_api_key
    from server.db.models import Tenant
    from server.db.session import get_engine
    from sqlalchemy.orm import sessionmaker

    db = sessionmaker(bind=get_engine())()
    db.add(Tenant(name="t1", api_key_hash=hash_api_key(api_key)))
    db.commit()
    db.close()
    return {"X-API-Key": api_key}


def test_avatar_lifecycle(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Alice"},
        files={"file": ("alice.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 201
    avatar_id = resp.json()["avatar_id"]

    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "ready"
    assert detail["config"]["mouth_scale"] == 1.0

    listing = client.get("/api/v1/avatars", headers=headers).json()
    assert [a["avatar_id"] for a in listing] == [avatar_id]

    assert client.delete(f"/api/v1/avatars/{avatar_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).status_code == 404


def test_avatar_tiny_image_becomes_failed(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "bad"},
        files={"file": ("bad.png", _png_bytes(50, 50), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "failed"
    assert detail["error"]


def test_avatar_not_found(client):
    headers = _seed_tenant(client)
    assert client.get("/api/v1/avatars/nope", headers=headers).status_code == 404


def test_upload_rejects_path_separator_in_suffix(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "evil"},
        files={"file": ("x.foo/bar", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json() == {
        "error": {
            "code": "invalid_filename",
            "message": "filename must not contain path separators or '..'",
            "detail": None,
        }
    }


def test_upload_rejects_backslash_dotdot_in_suffix(client):
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "evil"},
        files={"file": ("x.foo\\..\\bar", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_filename"


def test_validate_suffix_rejects_traversal_segments():
    from server.api.avatars import _validate_suffix
    from server.core.errors import AppError

    assert _validate_suffix(".png") == ".png"
    for bad in (".a/../b", ".a\\..\\b", "..", ".x/evil", ".x\\evil"):
        with pytest.raises(AppError) as ei:
            _validate_suffix(bad)
        assert ei.value.status == 400
        assert ei.value.code == "invalid_filename"


def test_derive_suffix_uses_last_extension_or_png():
    from server.api.avatars import _derive_suffix

    assert _derive_suffix("alice.png") == ".png"
    assert _derive_suffix(None) == ".png"
    assert _derive_suffix("noext") == ".png"
    assert _derive_suffix("x.y.JPEG") == ".JPEG"


def test_derive_suffix_rejects_traversal_filename():
    from server.api.avatars import _derive_suffix
    from server.core.errors import AppError

    with pytest.raises(AppError) as ei:
        _derive_suffix("x.png/../../evil")
    assert ei.value.status == 400
    assert ei.value.code == "invalid_filename"


def test_require_uuid_hex_rejects_non_hex():
    from server.api.avatars import _require_uuid_hex
    from server.core.errors import AppError

    assert _require_uuid_hex("a" * 32) == "a" * 32
    good = "0123456789abcdef0123456789abcdef"
    assert _require_uuid_hex(good) == good
    for bad in ("nope", "../evil", "..\\evil", "a" * 31, "a" * 33, "A" * 32, good + "0"):
        with pytest.raises(AppError) as ei:
            _require_uuid_hex(bad)
        assert ei.value.status == 404
        assert ei.value.code == "not_found"


def test_avatar_job_id_fits_pk_column_contract(client):
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from server.db.models import AvatarJob
    from server.db.session import get_engine

    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Alice"},
        files={"file": ("alice.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 201
    avatar_id = resp.json()["avatar_id"]
    db = sessionmaker(bind=get_engine())()
    try:
        job = db.execute(select(AvatarJob).where(AvatarJob.avatar_id == avatar_id)).scalars().one()
    finally:
        db.close()
    col_len = AvatarJob.__table__.c.id.type.length
    assert col_len == 32
    assert len(job.id) <= col_len


def test_delete_non_hex_avatar_id_is_404_and_skips_store(client, monkeypatch):
    headers = _seed_tenant(client)
    called = []

    def fake_delete(self, avatar_id):
        called.append(avatar_id)

    monkeypatch.setattr("server.avatar.store.AvatarStore.delete_assets", fake_delete)
    resp = client.delete("/api/v1/avatars/not-a-uuid", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
    assert called == []


def _stub_video(monkeypatch, duration=2.0, fps=25.0):
    # Both call sites look the names up on the module at call time (store does
    # `video.probe_video`, the mock pipeline `video.decode_frames`), which is
    # what makes patching `server.avatar.video` sufficient for both.
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": fps, "frame_count": int(duration * fps), "duration": duration},
    )
    import numpy as np

    monkeypatch.setattr(
        "server.avatar.video.decode_frames",
        lambda p, target_fps, max_frames, canvas: (
            np.full((1, canvas[1], canvas[0], 3), 128, np.uint8),
            fps,
        ),
    )


def test_avatar_video_lifecycle(client, monkeypatch):
    _stub_video(monkeypatch)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Video Alice"},
        files={"file": ("alice.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]

    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "ready"


def test_avatar_video_cap_counts_output_frames_not_source_frames(client, monkeypatch):
    # 15 s at 30 fps is 450 *source* frames but exactly 375 *output* frames at
    # 25 fps. The spec caps on duration-derived output frames, so this upload is
    # legal -- a cap on raw source frames would wrongly reject it. Without this
    # case, capping on `info["frame_count"]` passes every other test.
    _stub_video(monkeypatch, duration=15.0, fps=30.0)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "30fps"},
        files={"file": ("clip.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 201, resp.text


def test_avatar_video_too_long_is_422(client, monkeypatch):
    _stub_video(monkeypatch, duration=20.0)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Too long"},
        files={"file": ("long.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "video_too_long"


def test_avatar_undecodable_video_is_422(client, monkeypatch):
    from server.avatar.video import VideoError

    def boom(p):
        raise VideoError("cannot open video")

    monkeypatch.setattr("server.avatar.video.probe_video", boom)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Bad"},
        files={"file": ("bad.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_video"
