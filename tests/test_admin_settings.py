"""Admin-controlled tunables: a small key/value store driven by PATCH.

Settings live in their own SQLite table; the request body is a dict of
known keys with values; PATCH replaces any given set. Unknown keys and
out-of-range values are rejected with 400 so a typo never silently no-ops.
"""
from __future__ import annotations

def test_admin_settings_default_returns_defaults(client):
    resp = client.get("/api/admin/settings", )
    assert resp.status_code == 200
    body = resp.json()
    # Every default name must be present with a finite value.
    for key in (
        "face_crop_size",
        "bbox_shift",
        "min_face_width_px",
        "min_nose_ratio",
        "min_sharpness",
        "min_occupancy",
        "idle_crf",
        "idle_preset",
        "video_min_frames",
        "min_detection_confidence",
    ):
        assert key in body, f"missing default for {key}"
        assert body[key] is not None


def test_admin_settings_patch_known_key_persists(client):
    resp = client.patch(
        "/api/admin/settings",
        json={"face_crop_size": 320, "idle_crf": 16},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["face_crop_size"] == 320
    assert body["idle_crf"] == 16

    # A subsequent GET sees the override.
    resp = client.get("/api/admin/settings", )
    assert resp.json()["face_crop_size"] == 320


def test_admin_settings_patch_unknown_key_400(client):
    resp = client.patch(
        "/api/admin/settings",
        json={"not_a_real_param": 123},
    )
    assert resp.status_code == 400


def test_admin_settings_patch_out_of_range_400(client):
    # face_crop_size must be in [96, 512]
    resp = client.patch(
        "/api/admin/settings",
        json={"face_crop_size": 9},
    )
    assert resp.status_code == 400


def test_admin_settings_preset_must_be_known(client):
    resp = client.patch(
        "/api/admin/settings",
        json={"idle_preset": "warp_speed"},
    )
    assert resp.status_code == 400
