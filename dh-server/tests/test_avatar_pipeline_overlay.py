import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class _FakeDetector:
    """Stand-in for MediapipeLandmarkDetector used in this test only."""

    def __init__(self, task_path):
        self.task_path = task_path

    def detect(self, rgb):
        from dh_server.overlay.landmarks import Landmarks
        h, w = rgb.shape[:2]
        pts = np.zeros((468, 2), dtype=np.float32)
        for i in range(468):
            ang = 2 * np.pi * i / 468
            pts[i] = (w / 2 + 0.3 * w * np.cos(ang), h / 2 + 0.4 * h * np.sin(ang))
        return Landmarks(
            points=pts,
            eye_l_box=(int(w * 0.20), int(h * 0.30), int(w * 0.18), int(h * 0.10)),
            eye_r_box=(int(w * 0.62), int(h * 0.30), int(w * 0.18), int(h * 0.10)),
            mouth_box=(int(w * 0.32), int(h * 0.55), int(w * 0.36), int(h * 0.18)),
            cheek_l_box=(int(w * 0.18), int(h * 0.50), int(w * 0.14), int(h * 0.18)),
            cheek_r_box=(int(w * 0.68), int(h * 0.50), int(w * 0.14), int(h * 0.18)),
            confidence=0.9,
        )


def test_mock_pipeline_bakes_overlay_block_when_detector_succeeds(client, monkeypatch):
    """Mock pipeline should call prep.preload_overlays and write the overlay block."""
    # The bake needs opencv (the expression layers warp a patch with cv2),
    # and opencv-python has no cp314 wheel, so the torch-free `.venv`
    # cannot install it. The guard skips cleanly there instead of
    # erroring, as the torch-dependent files do.
    pytest.importorskip("cv2")
    from dh_server.overlay import landmarks as landmarks_mod
    monkeypatch.setattr(landmarks_mod, "MediapipeLandmarkDetector", _FakeDetector)

    # Also stub the task-path existence check by making the file path exist.
    # Easiest: write an empty file at the expected location.
    from dh_server.config import get_settings
    data_dir = Path(get_settings().data_dir)
    task_path = data_dir / "models" / "musetalk" / "face_landmarker.task"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_bytes(b"")

    buf = io.BytesIO()
    Image.new("RGB", (720, 1280), (60, 70, 80)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/avatars/{avatar_id}")
        if r.json()["status"] == "ready":
            break
        time.sleep(0.05)
    assert r.json()["status"] == "ready", r.text

    avatars_root = data_dir / "avatars" / avatar_id
    pack = json.loads((avatars_root / "pack.json").read_text())
    assert "overlay" in pack
    assert set(pack["overlay"]["tokens"]["expression"]) == {"neutral", "smile", "sad"}
    assert set(pack["overlay"]["tokens"]["gaze"]) == {"center", "left", "right"}
    assert len(list((avatars_root / "overlays").glob("*.png"))) == 10


def test_mock_pipeline_skips_overlay_when_task_file_missing(client, monkeypatch, tmp_path):
    """When face_landmarker.task is absent, pipeline continues without overlay."""
    from dh_server.config import get_settings
    task_path = Path(get_settings().data_dir) / "models" / "musetalk" / "face_landmarker.task"
    if task_path.exists():
        task_path.unlink()

    buf = io.BytesIO()
    Image.new("RGB", (720, 1280), (60, 70, 80)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        r = client.get(f"/api/v1/avatars/{avatar_id}")
        if r.json()["status"] == "ready":
            break
        time.sleep(0.05)
    avatars_root = Path(get_settings().data_dir) / "avatars" / avatar_id
    pack = json.loads((avatars_root / "pack.json").read_text())
    assert "overlay" not in pack


def test_write_overlay_block_merges_into_pack_json(tmp_path):
    """store.write_overlay_block writes/updates overlay section without nuking siblings."""
    from dh_server.engine.base import AvatarPack
    from dh_server.avatar.store import AvatarStore

    store = AvatarStore(tmp_path)
    av_dir = store.avatar_dir("a1")
    av_dir.mkdir(parents=True)
    (av_dir / "pack.json").write_text(json.dumps({"crop": {"mode": "center"}}))
    pack = AvatarPack(avatar_id="a1", root=av_dir,
                       source_image_path=tmp_path / "src.png", config={}, prep={})

    from dh_server.avatar.store import write_overlay_block
    write_overlay_block(pack, {"version": 1, "tokens": {"expression": ["neutral"]}})
    on_disk = json.loads((av_dir / "pack.json").read_text())
    assert on_disk["crop"] == {"mode": "center"}
    assert on_disk["overlay"]["version"] == 1
    assert on_disk["overlay"]["tokens"]["expression"] == ["neutral"]


def test_load_pack_surfaces_overlay_block(client, tmp_path, monkeypatch):
    """load_pack must put the overlay block into pack.prep so the runtime
    compositor sees it. Regression for review #1: previously the block
    was written to pack.json but never reached AvatarPack.prep, leaving
    every overlay compositor permanently disabled."""
    # The avatar build bakes the overlays, and the bake needs opencv --
    # see the note on the sibling test above.
    pytest.importorskip("cv2")
    from dh_server.overlay import landmarks as landmarks_mod
    monkeypatch.setattr(landmarks_mod, "MediapipeLandmarkDetector", _FakeDetector)

    from dh_server.config import get_settings
    task_path = Path(get_settings().data_dir) / "models" / "musetalk" / "face_landmarker.task"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_bytes(b"")

    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (720, 1280), (60, 70, 80)).save(buf, format="PNG")
    resp = client.post(
        "/api/v1/avatars",
        data={"name": "a"},
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    avatar_id = resp.json()["avatar_id"]
    import time
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if client.get(f"/api/v1/avatars/{avatar_id}").json()["status"] == "ready":
            break
        time.sleep(0.05)

    from dh_server.avatar.store import AvatarStore
    store = AvatarStore(Path(get_settings().data_dir))
    av_dir = store.avatar_dir(avatar_id)
    from dh_server.db.models import Avatar
    from sqlalchemy.orm import sessionmaker
    from dh_server.db.session import get_engine
    with sessionmaker(bind=get_engine())() as db:
        avatar_row = db.get(Avatar, avatar_id)
        pack = store.load_pack(avatar_row)
    assert "overlay" in pack.prep, pack.prep
    assert pack.prep["overlay"]["version"] == 1