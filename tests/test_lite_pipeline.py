"""The `lite` avatar pipeline: what it refuses and what it writes.

The visual checkpoint is personal to one person, so the pipeline insists on
being told which way to get one: train it here, or import one that was
trained elsewhere. Anything else is refused with a message that says so --
silently training for an hour would be worse.

Needs cv2 (to decode the source video), so it runs in the runtime venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from server.avatar.lite_pipeline import LiteAvatarPipeline  # noqa: E402
from server.avatar.pipeline import PipelineError  # noqa: E402
from server.lite import pack as lite_pack  # noqa: E402

WIDTH, HEIGHT = 720, 1280


class _StubDetector:
    """A face detector that always finds the same face."""

    def detect(self, frame):
        points = np.zeros((110, 2), dtype=np.int32)
        points[1] = (200, 300)
        points[31] = (400, 310)
        points[52] = (300, 250)
        return points


def _write_video(path: Path, frames: int = 6, fps: int = 25) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    for index in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 30 + index * 5
        writer.write(frame)
    writer.release()
    return path


def test_a_photo_is_refused(tmp_path):
    """The model replays the person's motion, so it needs a video."""
    photo = tmp_path / "source.png"
    cv2.imwrite(str(photo), np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))

    pipeline = LiteAvatarPipeline(detector=_StubDetector())

    with pytest.raises(PipelineError) as excinfo:
        pipeline.process("a1", photo, tmp_path / "out", lambda stage, pct: None)
    assert "视频" in str(excinfo.value)


def test_a_build_without_a_checkpoint_or_training_is_refused(tmp_path):
    """Silently training for an hour would be worse than saying no."""
    video = _write_video(tmp_path / "source.mp4")
    pipeline = LiteAvatarPipeline(detector=_StubDetector())

    with pytest.raises(PipelineError) as excinfo:
        pipeline.process("a1", video, tmp_path / "out", lambda stage, pct: None)
    assert "checkpoint" in str(excinfo.value)


def test_a_missing_checkpoint_is_refused(tmp_path):
    video = _write_video(tmp_path / "source.mp4")
    pipeline = LiteAvatarPipeline(detector=_StubDetector())

    with pytest.raises(PipelineError):
        pipeline.process(
            "a1",
            video,
            tmp_path / "out",
            lambda stage, pct: None,
            options={"checkpoint": str(tmp_path / "nope.pth")},
        )


def test_the_pack_is_written(tmp_path):
    video = _write_video(tmp_path / "source.mp4", frames=6)
    checkpoint = tmp_path / "visual.pth"
    checkpoint.write_bytes(b"not a real checkpoint, but the pipeline only copies it")
    pipeline = LiteAvatarPipeline(detector=_StubDetector())

    config = pipeline.process(
        "a1",
        video,
        tmp_path / "out",
        lambda stage, pct: None,
        options={"checkpoint": str(checkpoint)},
    )

    root = tmp_path / "out"
    assert config["model_type"] == "feathertalk"
    assert config["frame_count"] == 6
    assert config["fps"] == 25
    # The pack names its own files.
    assert config["coords"] == "lite/coords.json"
    assert config["visual"] == "lite/visual.pth"
    # The frames and the face boxes are there.
    frames = sorted(lite_pack.frames_dir(root).glob("*.jpg"))
    assert len(frames) == 6
    coords = lite_pack.read_coords(root)
    assert len(coords) == 6
    assert coords[0]["bbox"] == [200, 250, 400, 450]
    assert coords[0]["size"] == [200, 200]
    # The checkpoint was copied into the pack.
    assert lite_pack.visual_path(root).read_bytes() == checkpoint.read_bytes()
    # And the metadata says what the engine needs.
    meta = lite_pack.read_meta(root)
    assert meta["frame_count"] == 6
    assert meta["causal_audio"] is True
    # The work directory does not survive the build.
    assert not (root / "lite_work").exists()


def test_the_build_reports_progress(tmp_path):
    video = _write_video(tmp_path / "source.mp4", frames=4)
    checkpoint = tmp_path / "visual.pth"
    checkpoint.write_bytes(b"x")
    pipeline = LiteAvatarPipeline(detector=_StubDetector())
    stages: list[tuple[str, int]] = []

    pipeline.process(
        "a1",
        video,
        tmp_path / "out",
        lambda stage, pct: stages.append((stage, pct)),
        options={"checkpoint": str(checkpoint)},
    )

    assert stages
    assert stages[-1] == ("build_pack", 100)
    # The percentages never go backwards.
    percentages = [pct for _, pct in stages]
    assert percentages == sorted(percentages)
