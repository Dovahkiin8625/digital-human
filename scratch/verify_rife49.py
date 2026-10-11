"""Manual GPU-venv check: rife49 weights load the vendored net and interleave."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from rife.model import RifeModel  # noqa: E402

m = RifeModel(ROOT / "scratch" / "rife49.pth")
print("device:", m.device)

rng = np.random.default_rng(0)
H, W = 1280, 720
# Two smooth, distinct frames (random noise would be pure occlusion).
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
a = np.stack(
    [
        (xx / W * 200 + 40),
        (yy / H * 160 + 20),
        ((xx + yy) / (W + H) * 180 + 10),
    ],
    axis=-1,
).astype(np.uint8)
b = np.roll(a, 40, axis=1)
b = np.ascontiguousarray((b.astype(np.int16) + 30).clip(0, 255).astype(np.uint8))

t0 = time.time()
same = m.interpolate(a, a, 0.5)
t1 = time.time()
mid = m.interpolate(a, b, 0.5)
t2 = time.time()
early = m.interpolate(a, b, 0.1)
t3 = time.time()

print("shape/dtype:", same.shape, same.dtype)
print("identical frames max diff:", np.abs(same.astype(int) - a.astype(int)).max())
print("midpoint differs from a/b:", np.abs(mid.astype(int) - a.astype(int)).mean(),
      np.abs(mid.astype(int) - b.astype(int)).mean())
d_early = np.abs(early.astype(int) - a.astype(int)).mean()
d_mid = np.abs(mid.astype(int) - a.astype(int)).mean()
print(f"t=0.1 closer to a than t=0.5: {d_early:.2f} < {d_mid:.2f}")
print(f"times: {t1-t0:.3f}s (first), {t2-t1:.3f}s, {t3-t2:.3f}s")
assert same.shape == a.shape
# Bilinear flow resampling on a near-zero estimated flow is allowed a few
# uint8 levels; the clips themselves are H.264-lossy before RIFE sees them.
assert np.abs(same.astype(int) - a.astype(int)).max() <= 8, "identical inputs must reconstruct"
assert d_early < d_mid, "timestep direction is wrong"
print("OK")
