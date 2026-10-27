"""The closed-mouth idle track: the silence window and the baked frames.

The `lite` pack's frames come from a video of the person *speaking* -- that
video is the model's training material -- so the idle track cannot be the
motion frames themselves. It is derived: the same frames, repainted by the
model with a silent audio window, which closes the mouth.

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

from dh_server.engine.base import AvatarPack  # noqa: E402
from dh_server.engine.lite_engine import LiteEngine  # noqa: E402
from dh_server.lite import idle as lite_idle  # noqa: E402
from dh_server.lite import pack as lite_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "vendor" / "FeatherTalk" / "model.py"
if not VENDORED.exists():
    pytest.skip("FeatherTalk is not vendored; run scripts/setup_lite.sh")

WIDTH, HEIGHT = 720, 1280
FPS = 25
BBOX = [100, 150, 300, 350]
SIZE = [200, 200]


def _stand_in_model(root: Path):
    """A checkpoint the model loader accepts (random weights)."""
    from dh_server.lite import vendor
    from dh_server.lite.visual import VisualModel

    vendor.ensure_on_path()
    from model import create_model

    checkpoint = lite_pack.visual_path(root)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(create_model("feathertalk").state_dict(), checkpoint)
    return VisualModel.load(checkpoint, device="cpu")


def _write_pack(root: Path, frames: int = 3) -> AvatarPack:
    """A pack with real JPEGs and a stand-in visual checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    lite_pack.frames_dir(root).mkdir(parents=True, exist_ok=True)
    coords = []
    for index in range(frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 10  # BGR: a blue that shifts per frame
        cv2.imwrite(str(lite_pack.frame_path(root, index)), frame)
        coords.append({"bbox": BBOX, "size": SIZE})
    lite_pack.write_coords(root, coords)
    lite_pack.write_meta(root, {"version": lite_pack.VERSION, "frame_count": frames})
    _stand_in_model(root)
    block = lite_pack.config_block(frame_count=frames, fps=FPS, model_type="feathertalk")
    return AvatarPack(
        avatar_id="a1",
        root=root,
        source_image_path=root / "normalized.png",
        config={"lite": block},
        prep=json.loads(json.dumps(block)),
    )


def test_the_silence_window_is_one_silent_pair_repeated():
    """Silence has no past and no future: every frame carries the same pair.

    A video frame is 40 ms of audio and FeatherHuBERT emits one token per
    20 ms, so a frame is a *pair* of tokens -- the same pair in every slot
    of a silent window.
    """
    tokens = np.random.RandomState(0).randn(4, 1024).astype(np.float32)

    window = lite_idle.silence_window(tokens)

    assert window.shape == (40, 1024)
    pairs = window.reshape(20, 2, 1024)
    assert np.allclose(pairs, pairs[0:1])
    assert np.allclose(pairs[0, 0], tokens[0])


def test_the_silence_window_needs_features():
    with pytest.raises(ValueError):
        lite_idle.silence_window(np.zeros((0, 1024), dtype=np.float32))
    with pytest.raises(ValueError):
        lite_idle.silence_window(np.zeros((4, 512), dtype=np.float32))


def test_baking_writes_one_frame_per_motion_frame(tmp_path):
    root = tmp_path / "pack"
    pack = _write_pack(root)
    visual = _stand_in_model(root)
    window = lite_idle.silence_window(np.zeros((4, 1024), dtype=np.float32))

    written = lite_idle.bake_idle_frames(root, lite_pack.read_coords(root), visual, window)

    assert written == 3
    baked = sorted(lite_pack.idle_frames_dir(root).glob("*.jpg"))
    assert len(baked) == 3
    frame = cv2.imread(str(baked[0]))
    assert frame.shape == (HEIGHT, WIDTH, 3)
    # The face was repainted, so the idle frame is not the motion frame.
    source = cv2.imread(str(lite_pack.frame_path(root, 0)))
    assert not np.array_equal(frame[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]],
                              source[BBOX[1] : BBOX[3], BBOX[0] : BBOX[2]])


def test_a_half_written_idle_track_is_ignored(tmp_path):
    """The idle position and the speech position must be the same frame.

    A track that does not line up with the motion frames would make the
    avatar jump when it starts talking, so it is ignored.
    """
    root = tmp_path / "pack"
    pack = _write_pack(root, frames=3)
    visual = _stand_in_model(root)
    window = lite_idle.silence_window(np.zeros((4, 1024), dtype=np.float32))
    lite_idle.bake_idle_frames(root, lite_pack.read_coords(root), visual, window)
    # One frame too few.
    lite_pack.idle_frame_path(root, 2).unlink()

    engine = LiteEngine(fps=FPS, encoder=lambda waveform: np.zeros((0, 1024), dtype=np.float32))
    ctx = engine.new_context(pack)
    asyncio.run(engine.prepare(pack))

    assert asyncio.run(engine.idle_frame(pack, ctx, 0)) is None


def test_the_engine_serves_the_baked_track(tmp_path):
    """The idle frames the engine hands out are the baked ones."""
    root = tmp_path / "pack"
    pack = _write_pack(root)
    visual = _stand_in_model(root)
    window = lite_idle.silence_window(np.zeros((4, 1024), dtype=np.float32))
    lite_idle.bake_idle_frames(root, lite_pack.read_coords(root), visual, window)

    engine = LiteEngine(fps=FPS, encoder=lambda waveform: np.zeros((0, 1024), dtype=np.float32))
    ctx = engine.new_context(pack)
    asyncio.run(engine.prepare(pack))

    idle = asyncio.run(engine.idle_frame(pack, ctx, 0))

    assert idle is not None
    baked = cv2.imread(str(lite_pack.idle_frame_path(root, 0)))[:, :, ::-1]
    assert np.array_equal(idle, baked)
