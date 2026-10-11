"""Forward-loop sequence assembly for video avatars.

The frame sequence the engine plays must end where it begins, otherwise every
loop boundary is a visible jump. The base frames run forward (kept[0..N-1]),
the tail crossfades back into the head (up to K blend frames), and the
resulting sequence is encoded once into ``musetalk/base.mp4``.

A short clip (a one-shot clip or anything shorter than the chosen K) cannot
crossfade meaningfully, so the helper falls back to a single hard cut from
the tail into the head and emits a warning -- the operator sees the loop seam
instead of a number out of range.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

#: Crossfade length. 8 blend frames at 25 fps = 320 ms; long enough that the
#: eye reads it as motion, short enough that the bridge holds only the few
#: cycles the engine will reuse.
CROSSFADE_K = 8

#: Operator-facing message when the clip is shorter than the crossfade.
SHORT_CLIP_WARNING = "驱动视频时长过短，已直接拼接首尾帧，请使用 5s 以上的循环素材"


def _lerp_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel linear blend (RGB uint8). Numpy does this in one pass.

    Unbiased midpoint, not +0.5 biased -- the engine's reference is the
    MuseTalk re-decoded base.mp4, which itself is H.264-lossy, so a true
    average is what makes `base.mp4` and the per-frame check round-trip
    predictably.
    """
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    return np.clip((a32 + b32) / 2.0, 0.0, 255.0).astype(np.uint8)


def build_loop_sequence(
    frames: list[np.ndarray],
    crossfade_k: int = CROSSFADE_K,
    prepare: Callable[[np.ndarray], tuple[np.ndarray, list[int]]] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Assemble the playable loop from the kept base frames.

    `prepare` is the same per-frame landmark/bbox/latent pipeline the
    engine uses; it is called on every blend frame so the resulting pack
    carries the artifacts the engine needs to render that frame too. A
    blend frame whose prep raises (no face, etc.) is replaced with a hard
    cut to `frames[0]` -- the loop stays continuous, and a hard cut is
    rarer than the operator already expects.

    Returns ``(loop_frames, warnings)``. ``loop_frames[-1]`` equals
    ``loop_frames[0]`` (the crossfade always reaches the head) unless the
    clip was shorter than `crossfade_k`, in which case the tail simply
    snaps to the head and `warnings` carries `SHORT_CLIP_WARNING`.
    """
    n = len(frames)
    if n == 0:
        return [], []
    warnings: list[str] = []
    out: list[np.ndarray] = list(frames)
    if n <= crossfade_k:
        warnings.append(SHORT_CLIP_WARNING)
        return out, warnings

    head = frames[0]
    for k in range(crossfade_k, 0, -1):
        tail = frames[-k]
        blend = _lerp_frames(tail, head)
        if prepare is not None:
            try:
                blend, _ = prepare(blend)
            except Exception:
                # A bridge frame that fails prep (no face, etc.) gets the hard
                # cut: renderers see the head again, no broken artifact row.
                blend = head
        out.append(blend)
    return out, warnings


def forward_index(frame: int, n: int) -> int:
    """Wrap an absolute frame counter to a forward-only sequence of length n.

    Replaces the legacy ping-pong ``cycle_index`` for packs with a baked
    base.mp4: the clip plays 0..n-1, 0..n-1, ... with no reversal, so the
    2N-2 period becomes N.
    """
    if n <= 0:
        return 0
    return frame % n