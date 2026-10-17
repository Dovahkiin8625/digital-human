"""Single source of truth for the host's GPU memory budget.

Each portrait animator's ``meta()`` consults :func:`get_vram_gb` so the
operator sees a single consistent dropdown across all backends: ones the
host can run are clickable; ones it cannot are greyed with a one-line
hint pointing at the real failure mode.

The helper is intentionally cheap to call repeatedly; importing ``torch``
+ the single ``get_device_properties(0)`` query is sub-millisecond after
the first CUDA init. We deliberately do *not* pin to a specific device
id -- the operator only sees one GPU in this app.
"""
from __future__ import annotations


def get_vram_gb() -> float | None:
    """Return total GPU memory in GB, or ``None`` when unknown.

    Three reasons for None:
    * ``torch`` is not installed (CI hosts without GPU deps).
    * CUDA is not available (CPU-only box).
    * ``torch.cuda.get_device_properties(0)`` raised for any reason --
      we swallow the failure so the meta() chain doesn't 500.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:  # noqa: BLE001 -- cuda init race / driver fault surfaces as unknown
        return None
