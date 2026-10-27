"""RIFE frame-rate remap for sub-25 fps avatar sources.

A 24 fps driver decoded onto the 25 fps timeline by frame repeat visibly
stutters (the repeat pattern rotates 1,1,1,2,...). RIFE synthesises the
in-between frame at the exact fractional position, so the source plays at
natural speed with one genuine frame per 40 ms slot.

Two pieces live here:

* ``timeline_plan`` / ``interpolate_timeline`` are torch-free -- which source
  pair each output slot blends, and at what timestep, is plain arithmetic and
  has to be testable in the base venv.
* ``RifeInterpolator`` imports torch and the vendored RIFE 4.9 network lazily;
  it only exists in the GPU venv and only ever runs during avatar preparation,
  never on the render path.

Weights: data/models/rife/flownet.pkl (RIFE v4.9, timestep-capable). A missing
file is not a preparation error -- the caller falls back to nearest-frame
decoding and surfaces `SUB25_WARNING` to the operator.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: Output slots whose computed timestep lands within this fraction of a source
#: boundary pass the source frame through rather than calling RIFE at t~=0/1.
_EPS = 1e-6

#: Surfaced in the pack's warnings when a slow source had to be carried by
#: frame repetition because the RIFE weights are not installed.
SUB25_WARNING = (
    "素材帧率低于 25fps，服务器缺少 RIFE 插帧权重 "
    "(data/models/rife/flownet.pkl)，时间轴以重复帧方式生成；"
    "放置权重后重新上传可获得更平滑的动态"
)

#: Vendored network root (inference-only slice of RIFE 4.9, MIT licensed).
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "RIFE"


def rife_weights_path() -> Path:
    from server.config import get_settings

    return Path(get_settings().data_dir) / "models" / "rife" / "flownet.pkl"


def timeline_plan(
    n_src: int, src_fps: float, dst_fps: int, max_frames: int
) -> list[tuple[int, float]]:
    """Map each output frame onto a source-frame pair.

    Returns ``[(src_index, timestep), ...]``: output slot k shows the frame at
    time ``(src_index + timestep) / src_fps``. ``timestep == 0.0`` means the
    source frame passes through unchanged; otherwise it is the RIFE blend
    position strictly inside ``(src_index, src_index + 1)``. Slots that would
    read past the last source frame clamp to it, so a clip's tail stalls by at
    most one slot instead of failing the job.
    """
    if n_src <= 0:
        return []
    duration = n_src / float(src_fps)
    n_out = max(1, min(int(max_frames), int(round(duration * dst_fps))))
    rate = float(src_fps) / float(dst_fps)
    plan: list[tuple[int, float]] = []
    for k in range(n_out):
        pos = k * rate
        i = int(math.floor(pos + _EPS))
        t = pos - i
        if t < _EPS:
            i, t = min(i, n_src - 1), 0.0
        elif t > 1.0 - _EPS:
            i, t = i + 1, 0.0
        if i >= n_src - 1:
            # Wants frame i+1, which does not exist (or sits on the last one).
            i, t = n_src - 1, 0.0
        plan.append((i, float(t)))
    return plan


def interpolate_timeline(
    frames: np.ndarray, plan: list[tuple[int, float]], rife
) -> np.ndarray:
    """Render `plan` against source ``frames`` ``[N,H,W,3]`` uint8 RGB.

    `rife` is any object with ``interpolate(a, b, timestep) -> frame`` -- the
    real `RifeInterpolator` or a test double. Passthrough slots (t == 0) copy
    the source frame without touching the model.
    """
    out = np.empty((len(plan),) + frames.shape[1:], dtype=np.uint8)
    for k, (i, t) in enumerate(plan):
        if t == 0.0:
            out[k] = frames[i]
        else:
            out[k] = rife.interpolate(frames[i], frames[i + 1], t)
    return out


class RifeInterpolator:
    """Lazy torch wrapper around the vendored RIFE 4.9 flownet."""

    def __init__(self, weights_path: Path | None = None):
        import sys

        if str(VENDOR_DIR.parent) not in sys.path:
            # The vendored package imports itself as `rife.model`, so the
            # *parent* of vendor/RIFE goes on the path.
            sys.path.insert(0, str(VENDOR_DIR.parent))

        from rife.model import RifeModel

        path = Path(weights_path) if weights_path else rife_weights_path()
        self._model = RifeModel(path)

    def interpolate(self, a: np.ndarray, b: np.ndarray, timestep: float) -> np.ndarray:
        """One RGB uint8 HWC frame at position `timestep` in (a, b)."""
        return self._model.interpolate(a, b, float(timestep))

    def remap(self, frames: np.ndarray, plan: list[tuple[int, float]]) -> np.ndarray:
        """Render `plan` over ``frames`` [N,H,W,3] uint8 RGB, caching per-source
        padded tensors and encode features (a 24->25 timeline revisits pairs)."""
        prepared: dict[int, tuple] = {}

        def prepared_frame(i: int):
            hit = prepared.get(i)
            if hit is None:
                tensor, h, w = self._model.to_tensor(frames[i])
                hit = (tensor, h, w, self._model.encode(tensor))
                prepared[i] = hit
            return hit

        out = np.empty((len(plan),) + frames.shape[1:], dtype=np.uint8)
        with self._model.inference_context():
            for k, (i, t) in enumerate(plan):
                if t == 0.0:
                    out[k] = frames[i]
                    continue
                ta, h, w, f0 = prepared_frame(i)
                tb, _, _, f1 = prepared_frame(i + 1)
                out[k] = self._model.run(ta, tb, t, f0, f1, h, w)
        return out
