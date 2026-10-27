"""The on-disk layout of a `wav2lip` avatar pack.

A `wav2lip` pack is a directory next to the avatar's other assets::

    <avatar_dir>/wav2lip/
        frames/00000000.jpg       the source frames, already on the 720x1280 canvas
        idle_frames/00000000.jpg  the same frames, repainted with the mouth closed
        coords.json               the face box of each frame, in canvas pixels
        meta.json                 what the engine needs to know

There is **no per-avatar weight** in the pack. Wav2Lip is a single generic
model: the same checkpoint drives every avatar, so the pack carries only the
frames and the boxes.

``idle_frames`` is the closed-mouth track a session shows while it has
nothing to say (see :mod:`dh_server.wav2lip.idle`). It is optional: a pack
built before the track existed does not have one.

``pack.json`` carries a ``wav2lip`` block naming these paths, the same way the
MuseTalk pack carries a ``musetalk`` block.
"""
from __future__ import annotations

import json
from pathlib import Path

#: The subdirectory the pack lives in, relative to the avatar directory.
SUBDIR = "wav2lip"
#: The pack format version. Bumped when the layout changes incompatibly.
VERSION = "1"
#: The key the pack's block has in ``pack.json``.
CONFIG_KEY = "wav2lip"

FRAMES_DIR = "frames"
IDLE_FRAMES_DIR = "idle_frames"
COORDS_FILE = "coords.json"
META_FILE = "meta.json"


def pack_dir(root: Path) -> Path:
    return Path(root) / SUBDIR


def frames_dir(root: Path) -> Path:
    return pack_dir(root) / FRAMES_DIR


def frame_path(root: Path, index: int) -> Path:
    return frames_dir(root) / f"{index:08d}.jpg"


def idle_frames_dir(root: Path) -> Path:
    return pack_dir(root) / IDLE_FRAMES_DIR


def idle_frame_path(root: Path, index: int) -> Path:
    return idle_frames_dir(root) / f"{index:08d}.jpg"


def coords_path(root: Path) -> Path:
    return pack_dir(root) / COORDS_FILE


def meta_path(root: Path) -> Path:
    return pack_dir(root) / META_FILE


def write_coords(root: Path, coords: list[dict]) -> None:
    pack_dir(root).mkdir(parents=True, exist_ok=True)
    coords_path(root).write_text(json.dumps(coords))


def read_coords(root: Path) -> list[dict]:
    """The per-frame face boxes. Raises ``FileNotFoundError`` when absent."""
    path = coords_path(root)
    if not path.exists():
        raise FileNotFoundError(f"wav2lip pack has no {path.name}")
    return json.loads(path.read_text())


def write_meta(root: Path, meta: dict) -> None:
    pack_dir(root).mkdir(parents=True, exist_ok=True)
    meta_path(root).write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_meta(root: Path) -> dict:
    path = meta_path(root)
    if not path.exists():
        raise FileNotFoundError(f"wav2lip pack has no {path.name}")
    return json.loads(path.read_text())


def config_block(
    *, frame_count: int, fps: int, model_type: str = "wav2lip", idle_frames: bool = False, **extra
) -> dict:
    """The ``wav2lip`` block for ``pack.json``.

    Every path in the block is relative to the avatar directory, so the
    block needs no root of its own. ``idle_frames`` says whether the pack
    carries the closed-mouth track; when it does, the block names it.
    """
    block = {
        "dir": SUBDIR,
        "version": VERSION,
        "model_type": model_type,
        "frame_count": frame_count,
        "fps": fps,
        "frames": f"{SUBDIR}/{FRAMES_DIR}",
        "coords": f"{SUBDIR}/{COORDS_FILE}",
        "meta": f"{SUBDIR}/{META_FILE}",
    }
    if idle_frames:
        block["idle_frames"] = f"{SUBDIR}/{IDLE_FRAMES_DIR}"
    block.update(extra)
    return block
