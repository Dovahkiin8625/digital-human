"""The on-disk layout of a `lite` avatar pack.

A `lite` pack is a directory next to the avatar's other assets::

    <avatar_dir>/lite/
        frames/00000000.jpg   the source frames, already on the 720x1280 canvas
        coords.json           the face box of each frame, in canvas pixels
        visual.pth            the personalised visual checkpoint
        meta.json             what the engine needs to know

The frames are the *same* pixels the training saw: the pipeline writes the
canvas frames once and the trainer reads them back, so there is no
"training resolution" and "serving resolution" to drift apart.

``pack.json`` carries a ``lite`` block naming these paths, the same way the
MuseTalk pack carries a ``musetalk`` block.
"""
from __future__ import annotations

import json
from pathlib import Path

#: The subdirectory the pack lives in, relative to the avatar directory.
SUBDIR = "lite"
#: The pack format version. Bumped when the layout changes incompatibly.
VERSION = "1"
#: The key the pack's block has in ``pack.json``.
CONFIG_KEY = "lite"

FRAMES_DIR = "frames"
COORDS_FILE = "coords.json"
VISUAL_FILE = "visual.pth"
META_FILE = "meta.json"


def pack_dir(root: Path) -> Path:
    return Path(root) / SUBDIR


def frames_dir(root: Path) -> Path:
    return pack_dir(root) / FRAMES_DIR


def frame_path(root: Path, index: int) -> Path:
    return frames_dir(root) / f"{index:08d}.jpg"


def coords_path(root: Path) -> Path:
    return pack_dir(root) / COORDS_FILE


def visual_path(root: Path) -> Path:
    return pack_dir(root) / VISUAL_FILE


def meta_path(root: Path) -> Path:
    return pack_dir(root) / META_FILE


def write_coords(root: Path, coords: list[dict]) -> None:
    pack_dir(root).mkdir(parents=True, exist_ok=True)
    coords_path(root).write_text(json.dumps(coords))


def read_coords(root: Path) -> list[dict]:
    """The per-frame face boxes. Raises ``FileNotFoundError`` when absent."""
    path = coords_path(root)
    if not path.exists():
        raise FileNotFoundError(f"lite pack has no {path.name}")
    return json.loads(path.read_text())


def write_meta(root: Path, meta: dict) -> None:
    pack_dir(root).mkdir(parents=True, exist_ok=True)
    meta_path(root).write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_meta(root: Path) -> dict:
    path = meta_path(root)
    if not path.exists():
        raise FileNotFoundError(f"lite pack has no {path.name}")
    return json.loads(path.read_text())


def config_block(*, frame_count: int, fps: int, model_type: str, **extra) -> dict:
    """The ``lite`` block for ``pack.json``.

    Every path in the block is relative to the avatar directory, so the
    block needs no root of its own.
    """
    block = {
        "dir": SUBDIR,
        "version": VERSION,
        "model_type": model_type,
        "frame_count": frame_count,
        "fps": fps,
        "frames": f"{SUBDIR}/{FRAMES_DIR}",
        "coords": f"{SUBDIR}/{COORDS_FILE}",
        "visual": f"{SUBDIR}/{VISUAL_FILE}",
        "meta": f"{SUBDIR}/{META_FILE}",
    }
    block.update(extra)
    return block
