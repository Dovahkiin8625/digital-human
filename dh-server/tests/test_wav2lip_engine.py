"""The `wav2lip` drive engine: the contract the session layer depends on.

The engine is exercised with the real architecture but a *stand-in* model:
what is pinned here is the contract -- the frame count, the frame format, the
idle path, the admin row -- not the lip quality, which needs the released
checkpoint and a real face.

Needs cv2, so it runs in the runtime venv and skips elsewhere.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dh_server.engine.base import AvatarPack  # noqa: E402
from dh_server.engine.wav2lip_engine import Wav2LipEngine, cycle_index  # noqa: E402
from dh_server.wav2lip import pack as wav2lip_pack  # noqa: E402

WIDTH, HEIGHT = 720, 1280
FPS = 25
SAMPLE_RATE = 48000
BLOCK = 4800  # 100 ms


class _StandInModel:
    """A model that paints a flat face, so a test can see where it landed."""

    def predict(self, faces: np.ndarray, mels: np.ndarray) -> np.ndarray:
        assert faces.shape[0] == mels.shape[0]
        out = np.zeros((faces.shape[0], 96, 96, 3), dtype=np.float32)
        out[:, :, :] = 0.5
        return out


def _write_pack(root: Path, frames: int = 4, idle: bool = True) -> AvatarPack:
    """A pack with real JPEGs and a face box per frame."""
    root.mkdir(parents=True, exist_ok=True)
    wav2lip_pack.frames_dir(root).mkdir(parents=True, exist_ok=True)
    if idle:
        wav2lip_pack.idle_frames_dir(root).mkdir(parents=True, exist_ok=True)
    coords = []
    for index in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 10  # BGR: a blue that shifts per frame
        cv2.imwrite(str(wav2lip_pack.frame_path(root, index)), frame)
        if idle:
            idle_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            idle_frame[:, :, 1] = 90  # green: the idle track's own colour
            cv2.imwrite(str(wav2lip_pack.idle_frame_path(root, index)), idle_frame)
        coords.append({"bbox": [100, 150, 300, 350]})
    wav2lip_pack.write_coords(root, coords)
    wav2lip_pack.write_meta(root, {"version": wav2lip_pack.VERSION, "frame_count": frames})
    block = wav2lip_pack.config_block(frame_count=frames, fps=FPS, idle_frames=idle)
    return AvatarPack(
        avatar_id="a1",
        root=root,
        source_image_path=root / "normalized.png",
        config={"wav2lip": block},
        prep=json.loads(json.dumps(block)),
    )


def test_the_frame_grid_follows_the_audio(tmp_path):
    """100 ms of audio is two or three frames of 25 fps, and the average
    is 25 -- the same accumulator MuseTalk uses."""
    pack = _write_pack(tmp_path / "pack")
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    ctx = engine.new_context(pack)

    produced = 0
    for _ in range(10):
        frames = asyncio.run(
            engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx)
        )
        produced += len(frames)

    # Ten blocks of 100 ms is one second, so 25 frames.
    assert produced == 25


def test_the_frames_are_rgb_uint8_on_the_canvas(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    ctx = engine.new_context(pack)
    frames = asyncio.run(
        engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx)
    )
    assert frames
    for frame in frames:
        assert frame.image.shape == (HEIGHT, WIDTH, 3)
        assert frame.image.dtype == np.uint8


def test_the_painted_face_lands_on_the_face_box(tmp_path):
    """The stand-in model paints a flat grey, so the box is grey and the
    rest of the frame is not."""
    pack = _write_pack(tmp_path / "pack")
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    ctx = engine.new_context(pack)
    frames = asyncio.run(
        engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx)
    )
    frame = frames[0].image
    # The box is (100, 150) to (300, 350) in canvas pixels.
    assert tuple(frame[250, 200]) == (128, 128, 128)
    assert frame[20, 20, 0] != 128


def test_the_idle_track_is_served_while_the_session_is_silent(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    asyncio.run(engine.prepare(pack))
    frame = asyncio.run(engine.idle_frame(pack, None, 0))
    assert frame is not None
    assert frame.shape == (HEIGHT, WIDTH, 3)
    # The idle track's frames are green; the motion frames are blue.
    assert frame[:, :, 1].max() == 90


def test_a_pack_without_the_idle_track_falls_back_to_nothing(tmp_path):
    pack = _write_pack(tmp_path / "pack", idle=False)
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    asyncio.run(engine.prepare(pack))
    assert asyncio.run(engine.idle_frame(pack, None, 0)) is None


def test_the_idle_position_follows_the_session(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    ctx = engine.new_context(pack)
    asyncio.run(engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx))
    assert engine.idle_position(ctx) == ctx.emitted


def test_the_admin_row_describes_the_runtime(tmp_path):
    engine = Wav2LipEngine(fps=FPS, model=_StandInModel())
    row = engine.describe()
    assert row["name"] == "wav2lip"
    assert row["gpu"] is False
    assert row["models"][0]["id"] == "wav2lip_runtime"
    assert row["models"][0]["can_unload"] is True


def test_the_cycle_walks_forward_and_back():
    """0, 1, 2, 3, 2, 1, 0, 1, ... -- the source frames are played forward
    and then backward, so the loop is seamless."""
    assert cycle_index(0, 4) == 0
    assert cycle_index(3, 4) == 3
    assert cycle_index(4, 4) == 2
    assert cycle_index(5, 4) == 1
    assert cycle_index(6, 4) == 0
    assert cycle_index(7, 4) == 1
    assert cycle_index(2, 1) == 0
