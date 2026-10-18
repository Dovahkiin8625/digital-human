"""Live GPU memory snapshot returned by ``server.avatar.vram.get_vram_snapshot``.

The Models tab needs more than a single ``get_vram_gb`` answer to render a
useful progress bar; it needs the *currently* allocated / reserved figure so
the operator can see how much of the card the runtime is using *now*. This
is the missing piece and lives next to ``get_vram_gb`` to keep VRAM as one
coherent module surface.

Returns ``None`` whenever torch is missing or CUDA isn't available -- same
contract as ``get_vram_gb`` so the UI can degrade identically.
"""
from __future__ import annotations


def _gb(b: float) -> float:
    """Match the helper's GB rounding deterministically."""
    return round(b / 1024 ** 3, 3)


def test_snapshot_is_none_when_torch_missing(monkeypatch):
    # The full default suite runs without torch; a missing import must not crash
    # the route -- the page renders its own CPU placeholder when it sees null.
    import builtins as _bi

    real_import = _bi.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)
    from server.avatar import vram

    assert vram.get_vram_snapshot() is None


def test_snapshot_is_none_when_cuda_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from server.avatar import vram

    assert vram.get_vram_snapshot() is None


def test_snapshot_returns_live_memory_when_cuda_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 4 * 1024 ** 3
        name = "Stub GPU"

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _idx: int(1.0 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _idx: int(1.5 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _idx: int(1.2 * 1024 ** 3))
    # mem_get_info returns (free, total) in bytes.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _idx: (3.0 * 1024 ** 3, 4.0 * 1024 ** 3))

    from server.avatar import vram

    snap = vram.get_vram_snapshot()
    assert snap is not None
    assert snap["device"] == "cuda:0"
    assert snap["name"] == "Stub GPU"
    assert snap["total_gb"] == _gb(4 * 1024 ** 3)
    assert snap["allocated_gb"] == _gb(1.0 * 1024 ** 3)
    assert snap["reserved_gb"] == _gb(1.5 * 1024 ** 3)
    assert snap["peak_gb"] == _gb(1.2 * 1024 ** 3)
    # ``free_gb`` reads mem_get_info's *free*, not total - allocated; the
    # snapshot must use the OS-reported figure so the bar moves when the
    # runtime is released.
    assert snap["free_gb"] == _gb(3.0 * 1024 ** 3)
