"""Training the personalised visual model, and the causal dataset it needs.

The upstream dataset builds its audio window with the *real* future audio:
frames ``[i-10, i+10)`` around the current one. Rendering a live stream with
that window would show a mouth for audio that has not been spoken yet, so the
server trains against a **causal** window instead -- the future slots carry
the current frame's tokens, which is exactly what
:mod:`server.lite.audio` feeds the model at serving time.

Training and inference therefore see the same distribution, which is the
whole point: a model trained on the upstream window and served on the causal
one would be fed inputs it never saw during training.

Nothing is modified in the vendored upstream: the causal dataset is a
subclass, and the upstream training loop is reused by swapping the class it
instantiates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from server.lite import vendor

#: The tokens that describe the future in a 40-token window: everything after
#: the current frame's two tokens. (The fill itself lives in
#: :mod:`server.lite.causal_dataset`, which has to be a module of its own so
#: the DataLoader can pickle it.)


def write_landmarks(path: Path, points: np.ndarray) -> None:
    """Write a ``.lms`` file: one ``"x y"`` line per landmark."""
    lines = [f"{int(x)} {int(y)}" for x, y in points]
    path.write_text("\n".join(lines) + "\n")


def causal_dataset_class():
    """The causal dataset class, for the upstream training loop.

    The class lives in its own module because PyTorch's DataLoader spawns
    worker processes on Windows and a class defined inside a function cannot
    be pickled into them.
    """
    vendor.ensure_on_path()
    from server.lite.causal_dataset import CausalTemporalMouthRoiDataset

    return CausalTemporalMouthRoiDataset


def train_visual_model(
    dataset_dir: Path,
    save_dir: Path,
    *,
    epochs: int = 60,
    batchsize: int = 8,
    lr: float = 1e-3,
    save_every: int = 5,
    resume: str = "",
    allow_cpu: bool = False,
    progress=None,
) -> Path:
    """Train the visual model for one person and return the checkpoint path.

    Reuses the upstream training loop (losses, checkpointing, resume) and
    swaps in the causal dataset. ``dataset_dir`` must hold the upstream
    layout: ``full_body_img/``, ``landmarks/`` and ``aud_hu.npy``.

    ``progress`` is called with ``(stage, percent)`` so the avatar job can
    report movement; the upstream loop does not know about jobs, so the
    percentage is derived from the epoch.
    """
    vendor.ensure_on_path()
    import train as upstream_train

    dataset_cls = causal_dataset_class()
    original = upstream_train.TemporalMouthRoiDataset
    upstream_train.TemporalMouthRoiDataset = dataset_cls
    argv = [
        "train.py",
        "--dataset_dir",
        str(dataset_dir),
        "--save_dir",
        str(save_dir),
        "--epochs",
        str(epochs),
        "--batchsize",
        str(batchsize),
        "--lr",
        str(lr),
        "--save_every",
        str(save_every),
    ]
    if resume:
        argv += ["--resume", str(resume)]
    if allow_cpu:
        argv.append("--allow_cpu")

    saved = sys.argv
    sys.argv = argv
    try:
        if progress is not None:
            progress("train", 0)
        upstream_train.main()
    finally:
        sys.argv = saved
        upstream_train.TemporalMouthRoiDataset = original

    checkpoint = Path(save_dir) / "last.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"training finished without writing {checkpoint}; "
            f"look in {save_dir} for the epoch checkpoints"
        )
    return checkpoint
