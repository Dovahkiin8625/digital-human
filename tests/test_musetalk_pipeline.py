"""Stage 1 contract tests for the real MuseTalk avatar pipeline.

A stub preparer stands in for the torch-side models, so these run in the
torch-free `.venv` and still pin the artifact contract the engine consumes:
which files exist, where the coordinates live, and what `pack.json` carries.
The torch-specific half (that a real `torch.save`/`torch.load` round-trips, and
that a real photo yields a real face crop) belongs to the GPU-marked tests.
"""
import json

import numpy as np
import pytest
from PIL import Image

from server.avatar import musetalk_pipeline as mp
from server.avatar.musetalk_pipeline import (
    FACE_CROP,
    MuseTalkAvatarPipeline,
    fit_to_canvas,
)
from server.avatar.pipeline import MockAvatarPipeline, PipelineError, create_pipeline
from server.avatar.store import AvatarStore

SIDECARS = ["latents.pt", "coords.json", "meta.json"]


class StubPreparer:
    """Correctly-shaped artifacts, no torch, no GPU."""

    def __init__(self):
        self.seen_size = None

    def __call__(self, bgr):
        self.seen_size = bgr.shape[:2]
        return {
            "bbox": [100, 200, 300, 400],
            "landmarks": np.zeros((468, 2), dtype=np.float32),
            "face_crop_bgr": np.zeros((FACE_CROP, FACE_CROP, 3), dtype=np.uint8),
            "latents": np.zeros((1, 8, 32, 32), dtype=np.float32),
            "mask": np.full((300, 300), 255, dtype=np.uint8),
            "mask_crop_box": [10, 20, 310, 320],
        }


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """Pipeline + a stub preparer, with `latents.pt` written by numpy."""
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    from server.config import get_settings

    get_settings.cache_clear()

    written = {}

    def fake_save_latents(parts, path):
        arr = np.concatenate([np.asarray(p) for p in parts], axis=0)
        written["shape"] = tuple(arr.shape)
        with open(path, "wb") as fh:
            np.save(fh, arr)

    monkeypatch.setattr(mp, "_save_latents", fake_save_latents)
    stub = StubPreparer()
    src = tmp_path / "photo.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    return MuseTalkAvatarPipeline(preparer=stub), stub, src, tmp_path / "out", written


def test_pack_contract_is_complete(wired):
    pipeline, _, src, out, written = wired
    stages = []

    cfg = pipeline.process("a1", src, out, lambda s, p: stages.append((s, p)))

    assert stages == [("detect_face", 10), ("align", 40), ("normalize", 70), ("build_pack", 100)]

    sidecar = out / "musetalk"
    for name in SIDECARS:
        assert (sidecar / name).exists(), f"missing sidecar {name}"
    assert (sidecar / "masks" / "00000000.png").exists()

    assert written["shape"] == (1, 8, 32, 32), "one image frame -> latents [1,8,32,32]"

    with Image.open(out / "normalized.png") as img:
        assert img.size == (720, 1280)

    coords = json.loads((sidecar / "coords.json").read_text())
    assert len(coords) == 1
    assert coords[0]["bbox"] == [100, 200, 300, 400]
    assert coords[0]["mask_crop_box"] == [10, 20, 310, 320]

    meta = json.loads((sidecar / "meta.json").read_text())
    assert meta["version"] == "1.5"
    assert meta["crop"] == FACE_CROP
    assert meta["mode"] == "image"
    assert meta["frame_count"] == 1
    assert meta["kept"] == [0]

    # pack.json carries only the small "musetalk" block, and it must be JSON-clean
    assert list(cfg) == ["musetalk"]
    assert json.loads(json.dumps(cfg)) == cfg
    assert cfg["musetalk"]["coords"] == "musetalk/coords.json"
    assert cfg["musetalk"]["masks"] == "musetalk/masks"
    assert cfg["musetalk"]["bbox"] == [100, 200, 300, 400]


def test_detection_runs_on_the_canvas_not_the_raw_photo(wired):
    """Coordinates are stored in canvas space, so the engine never rescales."""
    pipeline, stub, src, out, _ = wired

    pipeline.process("a2", src, out, lambda s, p: None)

    assert stub.seen_size == (1280, 720), "preparer must receive the 720x1280 canvas"


def test_load_pack_exposes_prep_block(tmp_path):
    from server.db.models import Avatar

    store = AvatarStore(tmp_path / "data")
    d = store.avatar_dir("a3")
    d.mkdir(parents=True)
    (d / "pack.json").write_text(json.dumps({"musetalk": {"crop": 256, "bbox": [1, 2, 3, 4]}}))
    avatar = Avatar(id="a3", name="a", tenant_id="t", source_image_path=str(tmp_path / "s.png"))

    pack = store.load_pack(avatar)

    assert pack.prep == {"crop": 256, "bbox": [1, 2, 3, 4]}


def test_create_pipeline_honours_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_AVATAR_PIPELINE", "mock")
    from server.config import get_settings

    get_settings.cache_clear()
    assert isinstance(create_pipeline(), MockAvatarPipeline)

    monkeypatch.setenv("DH_AVATAR_PIPELINE", "musetalk")
    get_settings.cache_clear()
    assert isinstance(create_pipeline(), MuseTalkAvatarPipeline)

    monkeypatch.setenv("DH_AVATAR_PIPELINE", "nope")
    get_settings.cache_clear()
    with pytest.raises(PipelineError):
        create_pipeline()
    get_settings.cache_clear()


def test_musetalk_pipeline_rejects_tiny_image(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    from server.config import get_settings

    get_settings.cache_clear()
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(tiny)

    with pytest.raises(PipelineError, match="256"):
        MuseTalkAvatarPipeline(preparer=StubPreparer()).process(
            "a4", tiny, tmp_path / "out", lambda s, p: None
        )


def test_fit_to_canvas_letterboxes_without_distortion():
    wide = fit_to_canvas(Image.new("RGB", (2000, 500), (255, 0, 0)), 720, 1280)
    assert wide.size == (720, 1280)
    # a 4:1 source scaled into a 9:16 canvas leaves bars top and bottom
    assert wide.getpixel((360, 5)) == (16, 16, 24)
    assert wide.getpixel((360, 640)) == (255, 0, 0)

    tall = fit_to_canvas(Image.new("RGB", (500, 2000), (0, 255, 0)), 720, 1280)
    assert tall.size == (720, 1280)
    assert tall.getpixel((5, 640)) == (16, 16, 24)


def test_patch_avatar_config_preserves_musetalk_block(client):
    """`patch_avatar_config` rewrites pack.json; it must not drop the engine's block."""
    from sqlalchemy.orm import sessionmaker

    from server.api.deps import hash_api_key
    from server.db.models import Avatar
    from server.db.session import get_engine

    db = sessionmaker(bind=get_engine())()
    try:
        db.add(Avatar(id="keepme", name="a", tenant_id="t", status="ready",
                      source_image_path="unused.png",
                      config_json={"musetalk": {"crop": 256, "bbox": [1, 2, 3, 4]}}))
        db.commit()
    finally:
        db.close()

    resp = client.patch(
        "/api/admin/avatars/keepme/config",
        headers={"X-Admin-Token": "dev-admin-token"},
        json={"mouth_scale": 2.0},
    )
    assert resp.status_code == 200, resp.text

    db = sessionmaker(bind=get_engine())()
    try:
        avatar = db.get(Avatar, "keepme")
        assert avatar.config_json["musetalk"] == {"crop": 256, "bbox": [1, 2, 3, 4]}
        assert avatar.config_json["mouth_scale"] == 2.0
    finally:
        db.close()
