"""The video branch of the musetalk pipeline: decode, drop face-less frames,
record which decoded frames survived, and refuse a video with too few faces."""
import json

import numpy as np
import pytest
from PIL import Image

from server.avatar import musetalk_pipeline as mp
from server.avatar.musetalk_pipeline import FACE_CROP, MuseTalkAvatarPipeline
from server.avatar.pipeline import PipelineError

CANVAS = (720, 1280)


class StubPreparer:
    """Detects a face in every frame except the ones whose centre is black."""

    def __init__(self, blank=()):
        self.blank = set(blank)
        self.seen = 0

    def __call__(self, bgr):
        i = self.seen
        self.seen += 1
        if i in self.blank:
            raise PipelineError("no face detected in the uploaded image")
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
    return written, tmp_path


def _video(monkeypatch, n_frames):
    def fake_decode(path, target_fps, max_frames, canvas):
        # Distinct per-channel values, not a flat grey. The decoded canvas is RGB,
        # the preparer wants BGR, and normalized.png is written back as RGB. A flat
        # grey survives a channel swap unchanged, so only a channel-asymmetric
        # frame can catch a missing flip in _video_frames.
        frame = np.zeros((canvas[1], canvas[0], 3), np.uint8)
        frame[..., 0], frame[..., 1], frame[..., 2] = 10, 20, 30
        return np.broadcast_to(frame, (n_frames,) + frame.shape).copy(), 30.0

    monkeypatch.setattr("server.avatar.video.decode_frames", fake_decode)


def test_video_pack_records_kept_indices(wired, monkeypatch):
    written, tmp_path = wired
    _video(monkeypatch, 12)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)

    # frames 3 and 7 have no face and must be dropped
    prep = StubPreparer(blank={3, 7})
    cfg = MuseTalkAvatarPipeline(preparer=prep).process("v1", src, tmp_path / "out", lambda s, p: None)

    sidecar = tmp_path / "out" / "musetalk"
    coords = json.loads((sidecar / "coords.json").read_text())
    meta = json.loads((sidecar / "meta.json").read_text())

    assert meta["mode"] == "video"
    assert meta["frame_count"] == 10
    assert meta["kept"] == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    assert len(coords) == 10
    assert written["shape"] == (10, 8, 32, 32)
    assert len(list((sidecar / "masks").glob("*.png"))) == 10
    assert cfg["musetalk"]["source"] == "clip.mp4"
    assert cfg["musetalk"]["frame_count"] == 10
    # RGB in -> BGR for the preparer -> RGB out again. Dropping the flip in
    # _video_frames swaps these two channels and fails only here.
    resting = np.asarray(Image.open(tmp_path / "out" / "normalized.png"))
    assert tuple(resting[0, 0]) == (10, 20, 30)


def test_video_with_too_few_faces_is_rejected(wired, monkeypatch):
    _, tmp_path = wired
    _video(monkeypatch, 12)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)

    prep = StubPreparer(blank=set(range(5, 12)))  # only 5 usable frames < MIN_FRAMES
    with pytest.raises(PipelineError, match="no usable face frames"):
        MuseTalkAvatarPipeline(preparer=prep).process("v2", src, tmp_path / "out", lambda s, p: None)


def test_video_with_no_faces_at_all_reports_the_video_error(wired, monkeypatch):
    """Zero usable frames is still a *video* failure, not an image one."""
    _, tmp_path = wired
    _video(monkeypatch, 12)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)

    prep = StubPreparer(blank=set(range(12)))
    with pytest.raises(PipelineError, match="no usable face frames"):
        MuseTalkAvatarPipeline(preparer=prep).process("v3", src, tmp_path / "out", lambda s, p: None)
