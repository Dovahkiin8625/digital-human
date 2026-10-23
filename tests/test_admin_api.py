import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image


def _png(size=(640, 800)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _upload_avatar(client, name):
    return client.post(
        "/api/v1/avatars", data={"name": name},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]


def _admin_avatar(client, name="a", size=(640, 800)):
    return client.post(
        "/api/admin/avatars", data={"name": name},
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


def test_system_info(client):
    body = client.get("/api/admin/system").json()
    assert body["engine"]["name"] == "mock"
    assert body["pipeline"] == "mock"
    assert body["gpu"] is False


def test_avatar_config_endpoint_removed(client):
    # The mouth/expression sliders never reached the real MuseTalk engine, so
    # their endpoint was removed rather than left pretending to work.
    avatar_id = _upload_avatar(client, "a")

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/config", json={"mouth_scale": 1.5}
    )
    # The /config child route existed only for the sliders' PATCH; with it
    # gone, nothing matches the path at all.
    assert resp.status_code == 404


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
    avatar_id = _upload_avatar(client, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/normalized.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 50


def test_idle_mp4_is_generated_then_served_with_range(client, monkeypatch):
    avatar_id = _upload_avatar(client, "a")
    root = _make_video_mode(client, avatar_id)

    def fake_build(avatar_dir, source, settings, **_):
        assert source == avatar_dir / "source.mp4"
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("server.avatar.video.build_idle_video", fake_build)

    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.content == b"FAKEMP4-DATA"

    # A ranged read is served by FileResponse (206 Partial Content).
    rng = client.get(
        f"/api/admin/avatars/{avatar_id}/idle.mp4", headers={"Range": "bytes=0-3"}
    )
    assert rng.status_code == 206
    assert rng.content == b"FAKE"


def test_idle_mp4_build_failure_is_503(client, monkeypatch):
    from server.avatar.video import VideoError

    avatar_id = _upload_avatar(client, "a")
    _make_video_mode(client, avatar_id)

    def boom(avatar_dir, source, settings, **_):
        raise VideoError("decoder died")

    monkeypatch.setattr("server.avatar.video.build_idle_video", boom)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "idle_video_unavailable"


def test_idle_mp4_404_for_photo_avatar(client):
    avatar_id = _upload_avatar(client, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4")
    assert resp.status_code == 404


def test_preview_png_route_is_removed(client):
    avatar_id = _upload_avatar(client, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/preview.png")
    assert resp.status_code == 404


def test_jobs_listing(client):
    avatar_id = _upload_avatar(client, "a")
    jobs = client.get("/api/admin/jobs").json()
    assert isinstance(jobs, list)
    row = next(j for j in jobs if j["avatar_id"] == avatar_id)
    assert set(row) >= {"job_id", "avatar_id", "stage", "progress", "error", "created_at"}


@pytest.mark.parametrize("status", ["failed", "pending", "ready"])
def test_avatar_retry_allowed_from_any_status(client, status):
    avatar_id = _upload_avatar(client, "a")
    _set_status(avatar_id, status)

    resp = client.post(f"/api/admin/avatars/{avatar_id}/retry")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # A brand-new job row exists for this avatar (the retry background task ran).
    jobs = client.get("/api/admin/jobs").json()
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

    rows = client.get("/api/admin/avatars").json()
    row = next(r for r in rows if r["avatar_id"] == avatar_id)
    assert row["status"] == "pending"

    jobs = client.get("/api/admin/jobs").json()
    job = next(j for j in jobs if j["avatar_id"] == avatar_id)
    assert scheduled == [job["job_id"]]


def test_admin_get_avatar_detail_and_error(client):
    good = _upload_avatar(client, "good")
    detail = client.get(f"/api/admin/avatars/{good}").json()
    assert set(detail) == {
        "avatar_id",
        "name",
        "status",
        # The console's Debug / Chat / Avatars pages branch on this to pick a
        # renderer; it was missing from the admin serialiser while /api/v1
        # carried it, which made a Live2D avatar unselectable.
        "kind",
        "config",
        "error",
        "created_at",
    }
    assert detail["avatar_id"] == good
    # Pack contents + tunables audit; ``_effective_params`` is the new audit row.
    assert {"crop", "_effective_params"} <= set(detail["config"])
    assert detail["config"]["crop"] == {"mode": "center"}
    assert detail["error"] is None

    bad = _upload_avatar(client, "bad")
    # Force the small-image failure path:
    _set_status(bad, "failed")
    failed = client.get(f"/api/admin/avatars/{bad}").json()
    assert failed["status"] == "failed"


def test_admin_get_avatar_not_found(client):
    assert client.get(f"/api/admin/avatars/{uuid4().hex}").status_code == 404
    assert client.get("/api/admin/avatars/not-a-uuid").status_code == 404


def test_admin_create_avatar_persists_idle_overrides(client):
    # The create form posts ``idle_crf`` + ``idle_preset`` alongside the file.
    # The endpoint must validate, store them on the avatar row, and the
    # stored values must be readable back. Omitted / empty form fields must
    # be treated as "use the default" -- an empty dict, not a validation
    # error, so a non-tweaking operator keeps the existing UX.
    resp = _admin_avatar(client, name="with_tweaks")
    assert resp.status_code == 201
    avatar_id = resp.json()["avatar_id"]

    from sqlalchemy.orm import sessionmaker

    from server.db.models import Avatar
    from server.db.session import get_engine

    Session = sessionmaker(bind=get_engine())
    session = Session()
    try:
        row = session.get(Avatar, avatar_id)
        # No knobs posted -> empty overrides.
        assert row.tunable_overrides == {}
    finally:
        session.close()

    # Post with explicit knobs.
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "tweaked", "idle_crf": 14, "idle_preset": "veryslow"},
        files={"file": ("b.png", _png(), "image/png")},
    )
    assert resp.status_code == 201
    avatar_id = resp.json()["avatar_id"]

    session = Session()
    try:
        row = session.get(Avatar, avatar_id)
        assert row.tunable_overrides == {"idle_crf": 14, "idle_preset": "veryslow"}
    finally:
        session.close()


def test_admin_create_avatar_rejects_out_of_range_idle_crf(client):
    # Validation: a clearly bad crf (above libx264's 51 ceiling) is rejected
    # with 400, not stored and not silently capped.
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "bad", "idle_crf": 99},
        files={"file": ("b.png", _png(), "image/png")},
    )
    assert resp.status_code == 400


def test_admin_create_avatar_rejects_unknown_idle_preset(client):
    # idle_preset must be on the x264 enum; an unknown string is rejected
    # with 400 (PyAV would otherwise silently swap to a default and the
    # operator's choice would never reach the encoder).
    resp = client.post(
        "/api/admin/avatars",
        data={"name": "bad", "idle_preset": "warp_speed"},
        files={"file": ("b.png", _png(), "image/png")},
    )
    assert resp.status_code == 400


def test_admin_delete_avatar(client):
    from server.avatar.store import AvatarStore
    from server.config import get_settings

    avatar_id = _upload_avatar(client, "a")
    store = AvatarStore(get_settings().data_dir)
    assert store.avatar_dir(avatar_id).exists()

    resp = client.delete(f"/api/admin/avatars/{avatar_id}")
    assert resp.status_code == 204

    rows = client.get("/api/admin/avatars").json()
    assert all(r["avatar_id"] != avatar_id for r in rows)
    jobs = client.get("/api/admin/jobs").json()
    assert all(j["avatar_id"] != avatar_id for j in jobs)
    assert not store.avatar_dir(avatar_id).exists()


def test_admin_delete_avatar_not_found(client):
    assert client.delete(f"/api/admin/avatars/{uuid4().hex}").status_code == 404
    assert client.delete("/api/admin/avatars/not-a-uuid").status_code == 404


# ----- LivePortrait drivers -------------------------------------------------

def _seed_drivers_dir(monkeypatch, *stems):
    """Drop empty ``<stem>.mp4`` files so the registry has something to list."""
    import pickle

    from server.avatar import liveportrait
    from server.config import get_settings

    drivers = get_settings().data_dir / "liveportrait_drivers"
    drivers.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", drivers, raising=False)
    for stem in stems:
        (drivers / f"{stem}.mp4").write_bytes(b"\x00" * 2048)
        # Also drop a pickle companion so cached=True.
        (drivers / f"{stem}.pkl").write_bytes(
            pickle.dumps({"n_frames": 12, "motion": []})
        )
    return drivers


def test_list_liveportrait_drivers_returns_present_files(client, monkeypatch):
    # ``motion_a`` stands in for the pkl-only canonical slot previously held
    # by ``laugh`` (deleted: pkl-only entries have no .gif and are no longer
    # shipped as canonical drivers).
    _seed_drivers_dir(monkeypatch, "subtle", "motion_a")
    resp = client.get("/api/admin/liveportrait-drivers")
    assert resp.status_code == 200
    body = resp.json()
    names = [d["name"] for d in body["drivers"]]
    # pkl wins over mp4 for ``motion_a``; ``subtle`` still surfaces.
    assert "subtle" in names
    assert "motion_a" in names
    by_name = {d["name"]: d for d in body["drivers"]}
    assert by_name["motion_a"]["kind"] == "pkl"
    assert by_name["motion_a"]["cached"] is True
    assert by_name["subtle"]["kind"] == "pkl"  # both files present -> pkl wins
    # Stable ordering: canonical first.
    assert body["drivers"][0]["name"] in liveportrait_canonicals()


def liveportrait_canonicals():
    # Imported lazily so the test module doesn't pull the full chain.
    from server.avatar import liveportrait

    return tuple(liveportrait._DRIVER_SOURCES)


def test_delete_liveportrait_driver_removes_files(client, monkeypatch):
    drivers = _seed_drivers_dir(monkeypatch, "my_custom")
    resp = client.delete("/api/admin/liveportrait-drivers/my_custom")
    assert resp.status_code == 204
    assert not (drivers / "my_custom.mp4").exists()
    assert not (drivers / "my_custom.pkl").exists()


def test_delete_canonical_driver_is_rejected(client, monkeypatch):
    # ``subtle`` is in the canonical registry; the fetch script is the
    # only sanctioned way to (re)create it.
    _seed_drivers_dir(monkeypatch, "subtle")
    resp = client.delete("/api/admin/liveportrait-drivers/subtle")
    assert resp.status_code == 400
    assert "re-run" in resp.json()["error"]["message"].lower() or "fetch" in resp.json()["error"]["message"].lower()


def test_extract_liveportrait_driver_rejects_duplicate(client, monkeypatch):
    drivers = _seed_drivers_dir(monkeypatch, "subtle")
    # Tiny "video" payload; the endpoint will refuse before invoking
    # the heavy upstream pipeline because ``subtle`` already exists.
    files = {"video": ("clip.mp4", b"\x00" * 16, "video/mp4")}
    resp = client.post(
        "/api/admin/liveportrait-drivers/extract",
        data={"name": "subtle", "max_seconds": "0"},
        files=files,
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "driver_exists"


def test_extract_liveportrait_driver_rejects_bad_name(client, monkeypatch):
    _seed_drivers_dir(monkeypatch)
    files = {"video": ("clip.mp4", b"\x00" * 16, "video/mp4")}
    # Path-unsafe characters must be rejected; ``/`` would escape
    # the drivers dir and ``..`` is a directory-traversal attempt.
    resp = client.post(
        "/api/admin/liveportrait-drivers/extract",
        data={"name": "foo/bar", "max_seconds": "0"},
        files=files,
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_driver_name"


def test_extract_liveportrait_driver_accepts_chinese_name(client, monkeypatch):
    """Operator-supplied Chinese template names must round-trip through the API."""
    files = {"video": ("clip.mp4", b"\x00" * 16, "video/mp4")}
    resp = client.post(
        "/api/admin/liveportrait-drivers/extract",
        data={"name": "微笑模板", "max_seconds": "0"},
        files=files,
    )
    # The endpoint will then try to invoke the upstream pipeline; in the
    # test env that fails (no LivePortrait shim) with 503. The point of
    # this test is that validation does NOT reject the name itself --
    # we should get past the 400 and reach the 503 (or 502) only.
    assert resp.status_code != 400, resp.text
    assert resp.status_code in (502, 503)