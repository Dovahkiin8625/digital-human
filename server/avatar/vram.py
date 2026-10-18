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


def get_vram_snapshot() -> dict | None:
    """Live per-device GPU memory snapshot, or ``None`` when unknown.

    Returns ``{device, name, total_gb, allocated_gb, reserved_gb, free_gb, peak_gb}``
    in GB (rounded to three decimals). All four *live* counters are queried
    once each and may disagree slightly under load -- reserved >= allocated
    on the same timeline, and ``free_gb`` is sourced from
    ``cuda.mem_get_info`` (driver-reported) rather than ``total - allocated``
    so the bar shows what the OS thinks is free, not the accounting view.

    ``None`` on the same conditions as :func:`get_vram_gb` -- the UI keeps the
    two in lockstep, so when one is unknown the Models tab hides its progress
    bar in lockstep with the dropdown's "won't fit" warning.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        dev_idx = 0
        props = torch.cuda.get_device_properties(dev_idx)
        # ``mem_get_info`` raises on a few older drivers; fall back to the
        # obvious derivation so the page still has a number to render.
        try:
            free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            free_gb = free_b / 1024 ** 3
        except Exception:  # noqa: BLE001 -- driver-side unavailability
            free_gb = (props.total_memory - torch.cuda.memory_reserved(dev_idx)) / 1024 ** 3
        return {
            "device": f"cuda:{dev_idx}",
            "name": getattr(props, "name", "GPU"),
            "total_gb": round(props.total_memory / 1024 ** 3, 3),
            "allocated_gb": round(torch.cuda.memory_allocated(dev_idx) / 1024 ** 3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(dev_idx) / 1024 ** 3, 3),
            "peak_gb": round(torch.cuda.max_memory_allocated(dev_idx) / 1024 ** 3, 3),
            "free_gb": round(free_gb, 3),
        }
    except Exception:  # noqa: BLE001 -- any CUDA-side fault surfaces as unknown
        return None
