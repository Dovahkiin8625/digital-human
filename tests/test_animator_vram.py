"""Per-backend VRAM-fit flags surfaced via ``meta()``.

The 形象活化 dropdown currently lists every registered backend regardless
of the host's VRAM budget, so an operator on a 16 GB card sees a Wan 2.2
backend they can't actually run. The dispatcher refuses at animate()-time,
but a clean UX surfaces the "won't fit" state in the dropdown instead of
letting the operator click + wait + see a "this host has X GB" error
message.

This module tests two contracts:

1. Each backend's ``meta()`` returns ``vram_required_gb`` and
   ``fits_current_gpu`` so the page can decide which options to disable.
2. ``server.avatar.vram.get_vram_gb()`` is the single source of VRAM
   truth; the dispatcher's animate() path can also use it to do the
   live VRAM recheck.
"""
from __future__ import annotations

from unittest import mock


# ----- get_vram_gb() -----


def test_get_vram_gb_returns_none_when_torch_missing(monkeypatch):
    # torch may not be installed in CI; the helper must degrade gracefully
    # instead of raising -- ``None`` means "unknown, page should let the
    # operator try".
    import builtins as _bi

    real_import = _bi.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)
    from server.avatar import vram

    assert vram.get_vram_gb() is None


def test_get_vram_gb_returns_none_when_cuda_not_available(monkeypatch):
    # torch installs but the host has no CUDA device: an admin-only box
    # running on CPU-only or remote. The dispatcher can't run any of the
    # GPU-bound backends; ``None`` keeps the page from falsely disabling
    # them.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from server.avatar import vram

    # Force the module's import path to actually go through torch by
    # importing it fresh.
    assert vram.get_vram_gb() is None


def test_get_vram_gb_returns_total_memory_in_gb(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 12 * 1024 ** 3  # 12 GB

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    from server.avatar import vram

    assert vram.get_vram_gb() == 12.0


# ----- meta() carries the new keys -----


def test_liveportrait_meta_exposes_vram_required_and_fit(monkeypatch):
    # LivePortrait meta must declare its ~6 GB budget so the dropdown can
    # grey the option out on a host that doesn't have it (e.g. CI).
    import torch

    from server.avatar import liveportrait

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 4 * 1024 ** 3  # 4 GB, below LivePortrait's 6 GB

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    meta = liveportrait.LivePortraitAnimator().meta()
    assert meta["vram_required_gb"] == 6
    assert meta["fits_current_gpu"] is False


def test_wan22_meta_marks_fit_on_consumer_card(monkeypatch):
    # A 16 GB host: Wan 2.2 5B-bf16 runs at ~10 GB peak with offload, so
    # the dropdown must keep it clickable on a consumer card.
    import torch

    from server.avatar import wan22

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 16 * 1024 ** 3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    metas = {a.name: a.meta() for a in wan22.all_animators()}
    assert metas["wan22-i2v-5b-bf16"]["fits_current_gpu"] is True


def test_wan22_meta_marks_unfit_on_tiny_card(monkeypatch):
    # An 8 GB host falls below the ~10 GB peak + 2 GB headroom budget.
    import torch

    from server.avatar import wan22

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Props:
        total_memory = 8 * 1024 ** 3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())
    metas = {a.name: a.meta() for a in wan22.all_animators()}
    assert metas["wan22-i2v-5b-bf16"]["fits_current_gpu"] is False


def test_musetalk_silent_meta_marks_fits(monkeypatch):
    # MuseTalk silent reuses the project's main engine GPU budget; we
    # don't second-guess that here.``vram_required_gb`` is null and
    # ``fits_current_gpu`` is True so the dropdown keeps the option
    # clickable even on tight hosts.
    from server.avatar import musetalk_silent

    meta = musetalk_silent.MuseTalkSilentAnimator().meta()
    assert meta["vram_required_gb"] is None
    assert meta["fits_current_gpu"] is True


def test_meta_vram_null_on_cpu_only_host(monkeypatch):
    # If cuda isn't available, ``fits_current_gpu`` is None (unknown) so
    # the page keeps the options clickable and the *real* failure shows up
    # inline when the user clicks 生成 instead of being silently hidden.
    import torch

    from server.avatar import liveportrait
    from server.avatar import wan22

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    lp_meta = liveportrait.LivePortraitAnimator().meta()
    assert lp_meta["vram_required_gb"] == 6
    assert lp_meta["fits_current_gpu"] is None
    for a in wan22.all_animators():
        assert a.meta()["fits_current_gpu"] is None
