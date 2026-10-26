"""The `lite` pack and how the store finds it.

Two things are pinned here:

* the pack's layout -- the files the engine reads and the names
  ``pack.json`` records for them;
* the store's lookup: the pack's block is named after the *model* that
  wrote it, so a pack can never be read by the wrong engine.

Neither needs torch or cv2, so this runs in the torch-free ``.venv``.
"""
from __future__ import annotations

import json

from server.avatar.store import AvatarStore
from server.db.models import Avatar
from server.lite import pack as lite_pack


def test_the_config_block_names_every_file_the_engine_reads():
    block = lite_pack.config_block(frame_count=12, fps=25, model_type="feathertalk")

    assert block["dir"] == "lite"
    assert block["coords"] == "lite/coords.json"
    assert block["visual"] == "lite/visual.pth"
    assert block["frames"] == "lite/frames"
    assert block["frame_count"] == 12
    assert block["fps"] == 25


def test_the_coords_round_trip(tmp_path):
    coords = [{"bbox": [1, 2, 3, 4], "size": [2, 2]}]

    lite_pack.write_coords(tmp_path, coords)

    assert lite_pack.read_coords(tmp_path) == coords


def test_the_meta_round_trip(tmp_path):
    meta = {"version": lite_pack.VERSION, "frame_count": 3, "causal_audio": True}

    lite_pack.write_meta(tmp_path, meta)

    assert lite_pack.read_meta(tmp_path) == meta


def test_the_store_reads_the_block_of_the_models_own_name(tmp_path):
    """A pack is read through the model that wrote it."""
    store = AvatarStore(tmp_path / "data")
    root = store.avatar_dir("a1")
    root.mkdir(parents=True)
    (root / "pack.json").write_text(
        json.dumps(
            {
                "musetalk": {"crop": 256},
                "lite": {"frame_count": 7, "fps": 25},
            }
        )
    )

    musetalk = store.load_pack(
        Avatar(id="a1", name="a", source_image_path="s.png", drive_model="musetalk")
    )
    lite = store.load_pack(
        Avatar(id="a1", name="a", source_image_path="s.png", drive_model="lite")
    )

    assert musetalk.prep == {"crop": 256}
    assert lite.prep == {"frame_count": 7, "fps": 25}


def test_the_store_falls_back_to_the_default_model_for_an_unflushed_row(tmp_path):
    """An avatar that has not been written yet has no drive_model in Python;
    the column's default is what the database would say."""
    store = AvatarStore(tmp_path / "data")
    root = store.avatar_dir("a1")
    root.mkdir(parents=True)
    (root / "pack.json").write_text(json.dumps({"musetalk": {"crop": 256}}))

    pack = store.load_pack(Avatar(id="a1", name="a", source_image_path="s.png"))

    assert pack.prep == {"crop": 256}
