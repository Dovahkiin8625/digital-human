"""The `wav2lip` pack layout: what the engine reads back from disk."""
from __future__ import annotations

import json
from pathlib import Path

from dh_server.wav2lip import pack as wav2lip_pack


def test_the_pack_lives_in_its_own_directory(tmp_path: Path):
    assert wav2lip_pack.pack_dir(tmp_path) == tmp_path / "wav2lip"
    assert wav2lip_pack.frames_dir(tmp_path) == tmp_path / "wav2lip" / "frames"
    assert wav2lip_pack.idle_frames_dir(tmp_path) == tmp_path / "wav2lip" / "idle_frames"
    assert wav2lip_pack.coords_path(tmp_path) == tmp_path / "wav2lip" / "coords.json"
    assert wav2lip_pack.meta_path(tmp_path) == tmp_path / "wav2lip" / "meta.json"


def test_the_frame_names_are_zero_padded(tmp_path: Path):
    assert wav2lip_pack.frame_path(tmp_path, 0).name == "00000000.jpg"
    assert wav2lip_pack.frame_path(tmp_path, 42).name == "00000042.jpg"
    assert wav2lip_pack.idle_frame_path(tmp_path, 7).name == "00000007.jpg"


def test_the_coords_round_trip(tmp_path: Path):
    coords = [{"bbox": [10, 20, 110, 220]}, {"bbox": [12, 24, 112, 224]}]
    wav2lip_pack.write_coords(tmp_path, coords)
    assert wav2lip_pack.read_coords(tmp_path) == coords


def test_reading_absent_coords_raises(tmp_path: Path):
    try:
        wav2lip_pack.read_coords(tmp_path)
    except FileNotFoundError as exc:
        assert "coords.json" in str(exc)
    else:  # pragma: no cover -- the call must raise
        raise AssertionError("read_coords should raise on a pack without coords")


def test_the_meta_round_trips(tmp_path: Path):
    meta = {"version": wav2lip_pack.VERSION, "frame_count": 4, "fps": 25}
    wav2lip_pack.write_meta(tmp_path, meta)
    assert wav2lip_pack.read_meta(tmp_path) == meta


def test_the_config_block_names_every_path_relative_to_the_avatar(tmp_path: Path):
    block = wav2lip_pack.config_block(frame_count=12, fps=25, idle_frames=True)
    assert block["dir"] == "wav2lip"
    assert block["frame_count"] == 12
    assert block["fps"] == 25
    assert block["frames"] == "wav2lip/frames"
    assert block["coords"] == "wav2lip/coords.json"
    assert block["meta"] == "wav2lip/meta.json"
    assert block["idle_frames"] == "wav2lip/idle_frames"
    # The block is what goes into pack.json, so it has to be JSON.
    assert json.loads(json.dumps(block)) == block


def test_a_pack_without_the_idle_track_does_not_name_it(tmp_path: Path):
    block = wav2lip_pack.config_block(frame_count=1, fps=25)
    assert "idle_frames" not in block
