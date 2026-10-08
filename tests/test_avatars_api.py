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
