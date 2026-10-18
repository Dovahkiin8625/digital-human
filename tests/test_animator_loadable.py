"""Per-animator load/unload lifecycle for the Models tab.

The 形象活化 tab used to lazy-load each backend's pipeline on first
``animate()`` call, paying ~30 s on a cold cache. The Models tab now
owns a *separate* lifecycle: each animator has ``load()`` /
``unload()`` / ``is_loaded`` so the operator can pre-allocate (or
release) before kicking off a generation. ``animate()`` on a backend
that wasn't loaded first raises a clear ``AnimatorError`` so the
dispatcher surfaces a 503 with a hint at the Models tab -- not a silent
mid-frame crash.

Each backend's contract:

* ``is_loaded`` -- bool, false at construction.
* ``load()`` -- construct the heavy pipeline + flip the flag. Heavy +
  only runs once; idempotent (re-call with already loaded is a no-op).
* ``unload()`` -- drop the cached pipeline + cuda empty_cache. Flips
  the flag back to False.

The tests below verify the contract end-to-end on the real backend
classes -- the heavy pipeline construction itself stays behind
``_load_pipeline`` (overridable in tests).
"""
from __future__ import annotations


# ----- LivePortrait --------------------------------------------------------


def test_liveportrait_constructor_is_not_loaded():
    from server.avatar.liveportrait import LivePortraitAnimator

    a = LivePortraitAnimator()
    assert a.is_loaded is False


def test_liveportrait_load_flips_flag_and_calls_load_pipeline(monkeypatch, tmp_path):
    from server.avatar.liveportrait import LivePortraitAnimator

    a = LivePortraitAnimator()
    seen = {"calls": 0, "returned": None}

    def fake_load(self):
        seen["calls"] += 1
        seen["returned"] = object()
        return seen["returned"]

    monkeypatch.setattr(LivePortraitAnimator, "_load_pipeline", fake_load)
    a.load()
    assert a.is_loaded is True
    assert a._pipeline is seen["returned"]
    assert seen["calls"] == 1


def test_liveportrait_load_is_idempotent(monkeypatch):
    # A second load() while already loaded is a no-op: the cost of the
    # heavy pipeline construction makes re-running it a real surprise
    # in a long-lived process.
    from server.avatar.liveportrait import LivePortraitAnimator

    a = LivePortraitAnimator()

    def fake_load(self):
        return object()

    monkeypatch.setattr(LivePortraitAnimator, "_load_pipeline", fake_load)
    a.load()
    first = a._pipeline
    a.load()
    assert a._pipeline is first


def test_liveportrait_unload_drops_pipeline_and_cache(monkeypatch):
    from server.avatar.liveportrait import LivePortraitAnimator

    a = LivePortraitAnimator()
    a._loaded = True
    a._pipeline = object()

    # Spy on cuda cache flush; the helper does empty_cache() and is_safe
    # to call without torch.
    cleared = {"n": 0}

    def fake_empty():
        cleared["n"] += 1

    monkeypatch.setattr("server.avatar.liveportrait._try_empty_cuda_cache", fake_empty)
    a.unload()
    assert a.is_loaded is False
    assert a._pipeline is None
    assert cleared["n"] == 1


def test_liveportrait_animate_raises_when_not_loaded(monkeypatch, tmp_path):
    # The whole point of this refactor: when the operator hasn't loaded
    # the backend, the dispatch surface must error with a hint at the
    # Models tab -- NOT silently pay the 30 s cold-load + produce output.
    from server.avatar.animators import AnimatorError
    from server.avatar.liveportrait import LivePortraitAnimator

    a = LivePortraitAnimator()
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(AnimatorError) as ei:
        a.animate(src, prompt="", num_frames=10, steps=0, out_dir=tmp_path / "out")
    assert "not_loaded" in str(ei.value)
    assert "Models" in str(ei.value) or "加载" in str(ei.value)


# ----- Wan 2.2 -------------------------------------------------------------


def test_wan22_constructor_is_not_loaded():
    from server.avatar.wan22 import Wan22I2VAnimator, VARIANTS

    a = Wan22I2VAnimator(variant=next(iter(VARIANTS)))
    assert a.is_loaded is False


def test_wan22_load_flips_flag_and_calls_load_pipeline(monkeypatch, tmp_path):
    from server.avatar.wan22 import Wan22I2VAnimator, VARIANTS

    a = Wan22I2VAnimator(variant=next(iter(VARIANTS)))
    seen = {"calls": 0, "returned": object()}

    def fake_load(self):
        seen["calls"] += 1
        return seen["returned"]

    monkeypatch.setattr(Wan22I2VAnimator, "_load_pipeline", fake_load)
    a.load()
    assert a.is_loaded is True
    assert a._pipeline is seen["returned"]


def test_wan22_unload_drops_pipeline(monkeypatch):
    import torch

    from server.avatar.wan22 import Wan22I2VAnimator, VARIANTS

    a = Wan22I2VAnimator(variant=next(iter(VARIANTS)))
    a._loaded = True
    a._pipeline = object()
    cleared = {"n": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.__setitem__("n", cleared["n"] + 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    a.unload()
    assert a.is_loaded is False
    assert a._pipeline is None
    assert cleared["n"] == 1


def test_wan22_animate_raises_when_not_loaded(monkeypatch, tmp_path):
    from server.avatar.animators import AnimatorError
    from server.avatar.wan22 import Wan22I2VAnimator, VARIANTS

    a = Wan22I2VAnimator(variant=next(iter(VARIANTS)))
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(AnimatorError) as ei:
        a.animate(src, prompt="", num_frames=17, steps=20, out_dir=tmp_path / "out")
    assert "not_loaded" in str(ei.value)


# ----- helpers ------------------------------------------------------------


import pytest  # noqa: E402  -- pytest fixtures must come at module scope anyway


# Force the import order to match real usage: torch / mediapipe / etc. may
# surface as imports in the modules above; we keep them lazy so the
# default suite still runs.
