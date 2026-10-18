"""LivePortrait portrait animator for the 形象活化 tab.

The real LivePortrait inference is heavy (≈ 1.5 GB of weights + torch +
onnxruntime + insightface) and not realistic in CI, so the adapter is
constructed to accept an injection of ``_run_pipeline``. Each test stubs
that method and asserts only the contract the dispatcher relies on: meta
shape, validation bounds, an mp4 at ``<out_dir>/out.mp4``, and that the
animator registers itself into ``ANIMATORS`` on import.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from server.avatar import animators


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _tiny_driver_mp4(path, frames: int = 30) -> None:
    """Encode a real (small) mp4 so the animator's decode path can run.

    The fixture needs a *valid* clip that ``cv2.VideoCapture`` will accept;
    a zero-byte placeholder passes the existence check but blows up at
    decode time. Use the same PyAV libx264 path the rest of the codebase
    uses to keep the test's "real bytes" property.
    """
    from server.avatar.video import _encode_h264

    arr = np.full((frames, 1280, 720, 3), 100, dtype=np.uint8)
    _encode_h264(path, arr, fps=25, crf=28, preset="ultrafast")


@pytest.fixture()
def fake_lp(monkeypatch, tmp_path):
    """Return the LivePortraitAnimator with its heavy path stubbed out.

    The stub records the call and writes a synthetic frame stack to the
    output directory; downstream code (the h264 encoder) then sees real
    bytes that decode cleanly. Anything downstream of ``_run_pipeline``
    still runs, which keeps the test honest about the wire format.

    New (since load/unload lifecycle): the fixture flips ``_loaded``
    to True so tests that don't care about the load state can keep
    calling ``animate()`` without first wiring the Models tab
    lifecycle. Tests that DO exercise the lifecycle leave the flag
    alone.
    """
    # Place a real (tiny) driving clip where the animator looks for it.
    # The bundled asset dir lives under data_dir/liveportrait_drivers in
    # production; the fixture points the constant at ``tmp_path`` instead.
    driver_dir = tmp_path / "liveportrait_drivers"
    driver_dir.mkdir(parents=True, exist_ok=True)
    _tiny_driver_mp4(driver_dir / "subtle.mp4", frames=30)
    from server.avatar import liveportrait

    monkeypatch.setattr(
        liveportrait,
        "LIVE_PORTRAIT_DRIVERS_DIR",
        driver_dir,
        raising=False,
    )

    def stub_pipeline(self, source_image, driver_frames, *, num_frames, driver="subtle", out_dir):
        # Mirror the contract: produce a stack of RGB uint8 at canvas size.
        # The animator's wrapper then resizes / encodes it as before.
        arr = np.full((num_frames, 1280, 720, 3), 100, dtype=np.uint8)
        np.save(out_dir / "_frames.npy", arr)
        return arr

    monkeypatch.setattr(liveportrait.LivePortraitAnimator, "_run_pipeline", stub_pipeline)
    a = liveportrait.LivePortraitAnimator()
    # Mark loaded so ``animate()`` doesn't 503-not-loaded; tests that
    # explicitly verify the lifecycle use their own animator instance
    # and skip this flip.
    a._loaded = True
    a._pipeline = object()
    return a


def test_registration_on_import():
    # Side-effect import: the module installs a single backend into the
    # registry under the canonical name.
    from server.avatar import liveportrait  # noqa: F401 -- side effect

    assert "liveportrait" in animators.ANIMATORS
    assert animators.ANIMATORS["liveportrait"].name == "liveportrait"


def test_meta_carries_ui_keys():
    from server.avatar import liveportrait

    meta = liveportrait.LivePortraitAnimator().meta()
    # The ActivatePortrait.tsx page reads these exact keys; a rename here
    # breaks the dropdown rendering silently.
    for key in (
        "label_zh",
        "group_zh",
        "description",
        "vram_hint",
        "accepts_prompt",
        "accepts_num_frames",
        "accepts_steps",
    ):
        assert key in meta, f"LivePortraitAnimator.meta() must return {key!r}"
    # LivePortrait doesn't use a textual prompt -- the form disables the
    # prompt textarea when this backend is picked.
    assert meta["accepts_prompt"] is False
    assert meta["accepts_steps"] is False
    assert meta["accepts_num_frames"] is True
    # The page groups entries by ``group_zh``; this drives the OptGroup label.
    assert meta["group_zh"] in ("本地（无需联网）", "本地驱动", "本地")


def test_validate_accepts_wide_num_frames_range():
    from server.avatar import liveportrait

    a = liveportrait.LivePortraitAnimator()
    # LivePortrait clips num_frames to the driving clip's length, so the
    # contract is permissive: anything from 1 up to a generous cap passes.
    # The page's input form already clamps at this maximum via InputNumber.
    assert a.validate(prompt="ignored", num_frames=1, steps=99) == {}
    assert a.validate(prompt="ignored", num_frames=125, steps=99) == {}
    assert a.validate(prompt="ignored", num_frames=1000, steps=99) == {}
    # Zero or negative frame counts are a programming error -- reject.
    errs = a.validate(prompt="", num_frames=0, steps=10)
    assert "num_frames" in errs


def test_is_ready_false_when_drivers_missing(tmp_path, monkeypatch):
    from server.avatar import liveportrait

    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", tmp_path, raising=False)
    a = liveportrait.LivePortraitAnimator()
    assert a.is_ready is False


def test_is_ready_true_when_drivers_present(fake_lp):
    # The fixture populates the drivers dir AND stubs the pipeline; both
    # halves of ``is_ready`` light green.
    assert fake_lp.is_ready is True


def test_animate_writes_out_mp4(fake_lp, tmp_path):
    out_dir = tmp_path / "out"
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    fake_lp.animate(
        src, prompt="", num_frames=49, steps=10, out_dir=out_dir
    )
    # Contract: the artifact lives at ``out.mp4`` with bytes.
    assert (out_dir / "out.mp4").exists()
    assert (out_dir / "out.mp4").stat().st_size > 0


def test_animate_raises_animator_error_when_not_ready(tmp_path):
    from server.avatar import animators
    from server.avatar import liveportrait

    # No drivers, no stub: ``is_ready`` is False. The animator must surface
    # a clean AnimatorError pointing at the fetch script, not a generic
    # exception, so the admin endpoint can map it to a 502 with a hint.
    # Point ``LIVE_PORTRAIT_DRIVERS_DIR`` at an EMPTY dir so the
    # "drivers not present" branch in animate() trips (it checks
    # ``drivers.is_dir() and any(drivers.iterdir())``).
    empty = tmp_path / "no-drivers"
    empty.mkdir()
    a = liveportrait.LivePortraitAnimator.__new__(liveportrait.LivePortraitAnimator)
    # Bypass the Models-tab load gate (the lifecycle added in this
    # iteration): the existing "not ready" surface must still fire for
    # a backend that was *loaded* but is missing the on-disk drivers
    # it depends on.
    a._loaded = True
    a._pipeline = object()
    liveportrait.LivePortraitAnimator.LIVE_PORTRAIT_DRIVERS_DIR = empty
    a.LIVE_PORTRAIT_DRIVERS_DIR = empty
    with pytest.raises(animators.AnimatorError) as exc:
        a.animate(tmp_path / "src.png", prompt="", num_frames=10, steps=10, out_dir=tmp_path / "out")
    msg = str(exc.value).lower()
    assert "liveportrait" in msg or "weights" in msg or "driver" in msg


def test_animate_clamps_num_frames_to_driver_length(fake_lp, tmp_path, monkeypatch):
    # If the user asks for more frames than the driving clip carries,
    # LivePortrait must silently truncate -- not crash. We can't easily
    # measure the real clip's length without running the inference, so we
    # assert the stub's recorded num_frames equals whatever the animator
    # decided after clamping.
    out_dir = tmp_path / "out"
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    seen = {}

    def stub_pipeline(self, source_image, driver_frames, *, num_frames, driver="subtle", out_dir):
        seen["num_frames"] = num_frames
        return np.full((num_frames, 1280, 720, 3), 50, dtype=np.uint8)

    from server.avatar import liveportrait

    monkeypatch.setattr(liveportrait.LivePortraitAnimator, "_run_pipeline", stub_pipeline)
    fake_lp.animate(src, prompt="", num_frames=9999, steps=10, out_dir=out_dir)
    # Clamp to a sane upper bound; 9999 is an operator typo and the page
    # already constrains the input. We just need *some* clamp.
    assert seen["num_frames"] < 9999
    assert (out_dir / "out.mp4").exists()


# ----- driver registry ---------------------------------------------------

def _pickle_driver(driver_dir, stem, n_frames=10):
    """Drop a tiny ``<stem>.pkl`` motion template next to a placeholder mp4."""
    import pickle

    driver_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / f"{stem}.pkl").write_bytes(
        pickle.dumps({"n_frames": n_frames, "motion": []})
    )


def test_list_drivers_reports_canonical_and_extras(tmp_path, monkeypatch):
    from server.avatar import liveportrait

    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", tmp_path, raising=False)
    # Two canonical drivers (one mp4-only, one with a cached pkl) and
    # one user-uploaded extra that is NOT in the canonical registry.
    # ``motion_a`` is canonical; we drop a .pkl so it acts as the
    # pkl-only slot the test used to cover with ``laugh``.
    _tiny_driver_mp4(tmp_path / "subtle.mp4", frames=12)
    _pickle_driver(tmp_path, "motion_a", n_frames=8)
    _tiny_driver_mp4(tmp_path / "my_custom.mp4", frames=4)

    rows = liveportrait.list_drivers()
    by_name = {r["name"]: r for r in rows}

    assert by_name["subtle"]["canonical"] is True
    assert by_name["subtle"]["kind"] == "mp4"
    assert by_name["subtle"]["cached"] is False
    assert by_name["motion_a"]["canonical"] is True
    assert by_name["motion_a"]["kind"] == "pkl"  # .pkl preferred when present
    assert by_name["motion_a"]["cached"] is True
    # User-uploaded extras surface without forking the registry.
    assert by_name["my_custom"]["canonical"] is False
    assert by_name["my_custom"]["label_zh"] == "my_custom"
    # Order: canonical ids first, then extras alphabetically.
    assert [r["name"] for r in rows if r["canonical"]] == [
        n for n in liveportrait._DRIVER_SOURCES
        if n in by_name
    ]


def test_resolve_driver_prefers_pkl_over_mp4(tmp_path, monkeypatch):
    from server.avatar import liveportrait

    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", tmp_path, raising=False)
    _tiny_driver_mp4(tmp_path / "motion_a.mp4", frames=4)
    _pickle_driver(tmp_path, "motion_a", n_frames=8)

    resolved = liveportrait.resolve_driver("motion_a")
    assert resolved.suffix == ".pkl", "pkl must win when both are present"


def test_resolve_driver_missing_lists_available(tmp_path, monkeypatch):
    from server.avatar import animators
    from server.avatar import liveportrait

    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", tmp_path, raising=False)
    _tiny_driver_mp4(tmp_path / "subtle.mp4", frames=12)

    with pytest.raises(animators.AnimatorError) as exc:
        liveportrait.resolve_driver("nonexistent")
    msg = str(exc.value).lower()
    assert "nonexistent" in msg
    assert "subtle" in msg  # the available list surfaces for the UI


def test_animate_unknown_driver_raises(monkeypatch, tmp_path):
    """``animate(driver=...)`` propagates the registry's AnimatorError."""
    from server.avatar import animators
    from server.avatar import liveportrait

    # Empty drivers dir so ``animate`` doesn't take the empty-dir branch
    # and reach the resolve step.
    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    a = liveportrait.LivePortraitAnimator()
    a._loaded = True  # bypass Models-tab gate; this test exercises resolve_driver
    a._pipeline = object()

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)

    with pytest.raises(animators.AnimatorError):
        a.animate(src, prompt="", num_frames=10, steps=0, driver="does_not_exist", out_dir=tmp_path / "out")


def test_num_frames_zero_means_full_template_length(monkeypatch, tmp_path):
    """``num_frames <= 0`` must use the template's full frame count, not zero.

    Regression: a previous default of 49 silently truncated a 78-frame
    template to 49 because the operator never overrode the value. The
    contract now is that ``num_frames`` is an upper bound the user can
    shorten but not exceed; the default (0 / blank) is "use it all".
    """
    from server.avatar import liveportrait

    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    # Drop a pkl with a known frame count so the resolver picks it up.
    import pickle

    (driver_dir / "subtle.pkl").write_bytes(pickle.dumps({"n_frames": 78, "motion": []}))
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)

    a = liveportrait.LivePortraitAnimator()
    a._loaded = True  # bypass Models-tab gate (test focuses on num_frames logic)
    a._pipeline = object()
    seen: dict = {}

    def stub(self, source_image, driver_frames, *, num_frames, driver="subtle", out_dir):
        seen["num_frames"] = num_frames
        # Return exactly the requested count of synthetic frames.
        return np.full((num_frames, 1280, 720, 3), 50, dtype=np.uint8)

    monkeypatch.setattr(liveportrait.LivePortraitAnimator, "_run_pipeline", stub)

    out_dir = tmp_path / "out"
    a.animate(src, prompt="", num_frames=0, steps=0, driver="subtle", out_dir=out_dir)
    # The full template length (78) was passed to the pipeline, not zero.
    assert seen["num_frames"] == 78
    assert (out_dir / "out.mp4").exists()


def test_num_frames_still_caps_template_length(monkeypatch, tmp_path):
    """``num_frames > 0`` still acts as a ceiling (cannot exceed template)."""
    from server.avatar import liveportrait

    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    import pickle

    (driver_dir / "subtle.pkl").write_bytes(pickle.dumps({"n_frames": 78, "motion": []}))
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)

    a = liveportrait.LivePortraitAnimator()
    a._loaded = True  # bypass Models-tab gate
    a._pipeline = object()
    seen: dict = {}

    def stub(self, source_image, driver_frames, *, num_frames, driver="subtle", out_dir):
        seen["num_frames"] = num_frames
        return np.full((num_frames, 1280, 720, 3), 50, dtype=np.uint8)

    monkeypatch.setattr(liveportrait.LivePortraitAnimator, "_run_pipeline", stub)

    a.animate(src, prompt="", num_frames=500, steps=0, driver="subtle", out_dir=tmp_path / "out")
    # 500 requested -> clamped to 78 (template length).
    assert seen["num_frames"] == 78


def test_list_drivers_includes_frames_from_pkl(monkeypatch, tmp_path):
    """``list_drivers()`` surfaces the frame count read from the .pkl."""
    import pickle

    from server.avatar import liveportrait

    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    (driver_dir / "subtle.pkl").write_bytes(pickle.dumps({"n_frames": 78, "motion": []}))
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    rows = liveportrait.list_drivers()
    by_name = {r["name"]: r for r in rows}
    assert by_name["subtle"]["frames"] == 78


def test_list_drivers_reads_frames_sidecar_for_mp4(monkeypatch, tmp_path):
    """For mp4 drivers without a .pkl, the .frames sidecar is the source."""
    from server.avatar import liveportrait

    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    _tiny_driver_mp4(driver_dir / "motion_a.mp4", frames=4)
    # Stub the read so we don't depend on cv2 in this test.
    (driver_dir / "motion_a.frames").write_text("123", encoding="utf-8")
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    rows = liveportrait.list_drivers()
    by_name = {r["name"]: r for r in rows}
    assert by_name["motion_a"]["frames"] == 123


def test_write_frame_count_sidecar_roundtrips(monkeypatch, tmp_path):
    from server.avatar import liveportrait

    driver_dir = tmp_path / "drivers"
    driver_dir.mkdir()
    monkeypatch.setattr(liveportrait, "LIVE_PORTRAIT_DRIVERS_DIR", driver_dir, raising=False)

    liveportrait.write_frame_count_sidecar("my_clip", 200)
    assert (driver_dir / "my_clip.frames").read_text(encoding="utf-8").strip() == "200"
    # Non-positive counts are silently skipped (defensive).
    liveportrait.write_frame_count_sidecar("ignored", 0)
    assert not (driver_dir / "ignored.frames").exists()


# ----- Models tab integration -----------------------------------------------


def test_liveportrait_has_weights_dir_property(monkeypatch):
    """The Models tab's portrait-animator row needs ``weights_dir`` so the
    operator can see where the weights live (and copy the path).

    Today the row helper probes ``("weights_dir", "WEIGHTS_ROOT",
    "_weights_root")`` and LivePortrait has none of those — it builds the
    path inline inside ``_load_pipeline``. Without this property, the row
    shows ``weights_dir=None`` and the operator sees an empty path cell.
    """
    from pathlib import Path

    from server.avatar import liveportrait

    anim = liveportrait.LivePortraitAnimator()
    settings_data_dir = Path("/tmp/fake_data")
    monkeypatch.setattr(liveportrait, "get_settings", lambda: type("S", (), {"data_dir": settings_data_dir})())

    wd = anim.weights_dir
    assert wd is not None
    assert str(wd).endswith("liveportrait"), (
        f"weights_dir should live under data_dir/models/liveportrait; got {wd}"
    )
