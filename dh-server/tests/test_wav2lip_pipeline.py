"""The `wav2lip` avatar pipeline: the pack it writes.

The detector and the model are stand-ins: what is pinned here is the pack's
*shape* -- the frames, the face boxes, the idle track and the config block
the engine reads back -- not the lip quality.

Needs cv2, so it runs in the runtime venv and skips elsewhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dh_server.avatar.wav2lip_pipeline import Wav2LipAvatarPipeline  # noqa: E402
from dh_server.wav2lip import pack as wav2lip_pack  # noqa: E402

WIDTH, HEIGHT = 720, 1280


class _StandInDetector:
    """Finds the same box in every frame, or nothing at all."""

    def __init__(self, box=(100, 150, 300, 350), misses=()):
        self.box = box
        self.misses = set(misses)

    def detect(self, frames):
        out = []
        for index in range(len(frames)):
            out.append(None if index in self.misses else self.box)
        return out


class _StandInModel:
    """Paints a flat face."""

    def predict(self, faces, mels):
        assert faces.shape[0] == mels.shape[0]
        return np.full((faces.shape[0], 96, 96, 3), 0.5, dtype=np.float32)


def _photo(path: Path, size=(400, 600)) -> Path:
    from PIL import Image

    Image.new("RGB", size, (30, 60, 90)).save(path)
    return path


def test_a_photo_becomes_a_one_frame_pack(tmp_path):
    source = _photo(tmp_path / "face.png")
    out = tmp_path / "avatar"
    out.mkdir()
    pipeline = Wav2LipAvatarPipeline(detector=_StandInDetector(), model=_StandInModel())
    config = pipeline.process("a1", source, out, lambda stage, pct: None)

    assert config["dir"] == "wav2lip"
    assert config["frame_count"] == 1
    assert config["fps"] == 25
    assert config["base_image"] == "normalized.png"
    assert config["idle_frames"] == "wav2lip/idle_frames"

    frames = sorted(wav2lip_pack.frames_dir(out).glob("*.jpg"))
    assert len(frames) == 1
    idle = sorted(wav2lip_pack.idle_frames_dir(out).glob("*.jpg"))
    assert len(idle) == 1
    assert (out / "normalized.png").exists()


def test_the_coords_name_the_face_box(tmp_path):
    source = _photo(tmp_path / "face.png")
    out = tmp_path / "avatar"
    out.mkdir()
    pipeline = Wav2LipAvatarPipeline(detector=_StandInDetector(), model=_StandInModel())
    pipeline.process("a1", source, out, lambda stage, pct: None)
    coords = wav2lip_pack.read_coords(out)
    assert coords == [{"bbox": [100, 150, 300, 350]}]


def test_a_source_with_no_face_is_refused(tmp_path):
    source = _photo(tmp_path / "face.png")
    out = tmp_path / "avatar"
    out.mkdir()
    pipeline = Wav2LipAvatarPipeline(detector=_StandInDetector(box=None), model=_StandInModel())
    with pytest.raises(Exception) as exc:
        pipeline.process("a1", source, out, lambda stage, pct: None)
    assert "人脸" in str(exc.value)


def test_the_idle_track_is_the_mouth_closed(tmp_path):
    """The idle frames are the motion frames repainted with a silent mel
    window, so they differ from the motion frames."""
    source = _photo(tmp_path / "face.png")
    out = tmp_path / "avatar"
    out.mkdir()
    pipeline = Wav2LipAvatarPipeline(detector=_StandInDetector(), model=_StandInModel())
    pipeline.process("a1", source, out, lambda stage, pct: None)
    motion = cv2.imread(str(wav2lip_pack.frame_path(out, 0)))
    idle = cv2.imread(str(wav2lip_pack.idle_frame_path(out, 0)))
    assert motion.shape == idle.shape
    assert not np.array_equal(motion, idle)
