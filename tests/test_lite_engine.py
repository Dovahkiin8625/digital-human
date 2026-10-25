"""The `lite` drive engine: the contract the session layer depends on.

The engine is exercised with the real architecture but a *stand-in*
checkpoint (random weights): what is pinned here is the contract -- the
frame count, the frame format, the idle path, the admin row -- not the lip
quality, which needs a checkpoint trained on a real person.

Needs torch, cv2 and the vendored FeatherTalk checkout, so it runs in the
runtime venv and skips elsewhere.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from server.engine.base import AvatarPack  # noqa: E402
from server.engine.lite_engine import LiteEngine, cycle_index  # noqa: E402
from server.lite import pack as lite_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "vendor" / "FeatherTalk" / "model.py"
if not VENDORED.exists():
    pytest.skip("FeatherTalk is not vendored; run scripts/setup_lite.sh")

WIDTH, HEIGHT = 720, 1280
FPS = 25
SAMPLE_RATE = 48000
BLOCK = 4800  # 100 ms


def _write_pack(root: Path, frames: int = 4) -> AvatarPack:
    """A pack with real JPEGs and a stand-in visual checkpoint."""
    from server.lite import vendor

    vendor.ensure_on_path()
    from model import create_model

    root.mkdir(parents=True, exist_ok=True)
    lite_pack.frames_dir(root).mkdir(parents=True, exist_ok=True)
    coords = []
    for index in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 10  # BGR: a blue that shifts per frame
        cv2.imwrite(str(lite_pack.frame_path(root, index)), frame)
        coords.append({"bbox": [100, 150, 300, 350], "size": [200, 200]})
    lite_pack.write_coords(root, coords)
    lite_pack.write_meta(root, {"version": lite_pack.VERSION, "frame_count": frames})
    torch.save(create_model("feathertalk").state_dict(), lite_pack.visual_path(root))
    return AvatarPack(
        avatar_id="a1",
        root=root,
        source_image_path=root / "normalized.png",
        config={"lite": lite_pack.config_block(root, frame_count=frames, fps=FPS, model_type="feathertalk")},
        prep=json.loads(json.dumps(lite_pack.config_block(root, frame_count=frames, fps=FPS, model_type="feathertalk"))),
    )


def _encoder(_waveform: np.ndarray) -> np.ndarray:
    """A stand-in audio encoder: one feature frame per 320 samples."""
    count = max(0, (len(_waveform) - 80) // 320)
    return np.zeros((count, 1024), dtype=np.float32)


def test_the_frame_grid_follows_the_audio(tmp_path):
    """100 ms of audio is two or three frames of 25 fps, and the average
    is 25 -- the same accumulator MuseTalk uses."""
    pack = _write_pack(tmp_path / "pack")
    engine = LiteEngine(fps=FPS, encoder=_encoder)
    ctx = engine.new_context(pack)

    produced = 0
    for _ in range(10):
        frames = asyncio.run(engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx))
        produced += len(frames)

    # Ten blocks of 100 ms is one second, so 25 frames.
    assert produced == 25


def test_the_frames_are_rgb_uint8_on_the_canvas(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = LiteEngine(fps=FPS, encoder=_encoder)
    ctx = engine.new_context(pack)

    frames = asyncio.run(engine.drive(pack, np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE, context=ctx))

    assert frames
    for frame in frames:
        assert frame.image.shape == (HEIGHT, WIDTH, 3)
        assert frame.image.dtype == np.uint8


def test_the_idle_path_returns_a_source_frame(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = LiteEngine(fps=FPS, encoder=_encoder)
    ctx = engine.new_context(pack)

    idle = asyncio.run(engine.idle_frame(pack, ctx, 1))

    assert idle is not None
    assert idle.shape == (HEIGHT, WIDTH, 3)


def test_the_idle_path_returns_none_when_the_pack_is_broken(tmp_path):
    pack = _write_pack(tmp_path / "pack")
    engine = LiteEngine(fps=FPS, encoder=_encoder)
    broken = AvatarPack(
        avatar_id="nope", root=Path("no-such-pack"), source_image_path=Path("x"), config={}, prep={}
    )

    assert asyncio.run(engine.idle_frame(broken, None, 0)) is None


def test_the_ping_pong_visits_every_frame_and_comes_back():
    assert [cycle_index(i, 4) for i in range(7)] == [0, 1, 2, 3, 2, 1, 0]
    assert cycle_index(0, 1) == 0


def test_the_admin_row_describes_the_runtime_without_loading_anything():
    # No encoder injected and no weights loaded: the row must report an
    # unloaded runtime rather than reaching for a weight file.
    engine = LiteEngine(fps=FPS)

    described = engine.describe()

    assert described["gpu"] is False
    assert described["loaded"] is False
    row = described["models"][0]
    assert row["kind"] == "drive_runtime"
    assert row["can_unload"] is False
