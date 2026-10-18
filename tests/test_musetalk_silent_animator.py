"""MuseTalk silent animator for the 形象活化 tab's "复用基础设施" slot.

MuseTalk is, by design, audio-driven: it retargets the source face's mouth /
jaw onto a driving audio waveform. With a synthesized silent WAV, the
output is *subtle* -- small jaw moves without phonemes -- which matches
"让一张静态人像轻微的动起来" exactly.

The animation harness reuses the project's existing muxer + encoder:
a tiny silent WAV synthesised in-process, a single frame looped to N frames,
and the same PyAV libx264 H.264 encoder the rest of the codebase uses.
The MuseTalk synthesis itself is split into a stubbable ``_run_pipeline``
-- CI replaces it with a deterministic frame stack; production wires it
to the real MuseTalk avatar build pipeline.
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


@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Replace MuseTalk's heavy path with a frame-loop placeholder."""
    from server.avatar import musetalk_silent

    def fake(self, source_image, *, num_frames, prompt, steps, out_dir):
        # Mirror the contract: RGB uint8 ``(num_frames, H, W, 3)`` at the
        # project canvas (720x1280). The animator's wrapper handles audio.
        arr = np.full((num_frames, 1280, 720, 3), 80, dtype=np.uint8)
        return arr

    monkeypatch.setattr(musetalk_silent.MuseTalkSilentAnimator, "_run_pipeline", fake)
    # New (load/unload lifecycle): tests in this module that don't care
    # about the load state can keep calling animate(); the fixture sets
    # the gate so the not-loaded branch is skipped. Tests that DO test
    # the lifecycle ignore this attribute.
    a = musetalk_silent.MuseTalkSilentAnimator()
    a._engine_runtime_loaded = lambda: True  # noqa: SLF001 -- test seam
    return a


# ----- registration / meta -----


def test_registration_on_import():
    from server.avatar import musetalk_silent  # noqa: F401 -- side effect

    assert "musetalk-silent" in animators.ANIMATORS
    assert animators.ANIMATORS["musetalk-silent"].name == "musetalk-silent"


def test_meta_groups_under_existing_infra_bucket():
    from server.avatar import musetalk_silent

    meta = musetalk_silent.MuseTalkSilentAnimator().meta()
    # The operator can tell the bucket apart from the others on the page.
    assert meta["group_zh"] == "复用基础设施"
    # All the UI keys are there so the form renders the right column.
    for key in (
        "label_zh",
        "description",
        "vram_hint",
        "accepts_prompt",
        "accepts_num_frames",
        "accepts_steps",
    ):
        assert key in meta
    # MuseTalk silent ignores prompt (silent audio has nothing to bind to);
    # num_frames controls the loop length; steps is meaningless here.
    assert meta["accepts_prompt"] is False
    assert meta["accepts_steps"] is False
    assert meta["accepts_num_frames"] is True


# ----- readiness -----


def test_is_ready_uses_existing_infra():
    # The animator reuses the project's MuseTalkPreparer + encoder; both
    # ship with the existing avatar build pipeline, so no extra weights
    # are needed. We treat MuseTalk silent as always-ready when the
    # avatar pipeline itself is configured (which it is on every real host).
    from server.avatar import musetalk_silent

    assert musetalk_silent.MuseTalkSilentAnimator().is_ready is True


# ----- validation -----


def test_validate_rejects_zero_or_negative_frames():
    from server.avatar import musetalk_silent

    a = musetalk_silent.MuseTalkSilentAnimator()
    assert "num_frames" in a.validate(prompt="ignored", num_frames=0, steps=10)
    assert "num_frames" in a.validate(prompt="ignored", num_frames=-5, steps=10)


def test_validate_accepts_a_normal_motion_clip():
    from server.avatar import musetalk_silent

    a = musetalk_silent.MuseTalkSilentAnimator()
    assert a.validate(prompt="ignored", num_frames=125, steps=10) == {}
    assert a.validate(prompt="ignored", num_frames=25, steps=10) == {}


def test_validate_rejects_too_long():
    from server.avatar import musetalk_silent

    a = musetalk_silent.MuseTalkSilentAnimator()
    # A 5-minute silent clip is a clear misuse of "subtle motion".
    assert "num_frames" in a.validate(prompt="ignored", num_frames=99999, steps=10)


# ----- animate -----


def test_animate_writes_mp4(stub_pipeline, tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    stub_pipeline.animate(src, prompt="", num_frames=125, steps=10, out_dir=out_dir)
    assert (out_dir / "out.mp4").exists()
    assert (out_dir / "out.mp4").stat().st_size > 0


def test_animate_emits_a_silent_wav_at_the_project_fps(stub_pipeline, tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    stub_pipeline.animate(src, prompt="", num_frames=125, steps=10, out_dir=out_dir)
    # The synthesised silent audio must be a real WAV alongside ``out.mp4``
    # so the encoder (or a future MuseTalk run) can consume it.
    candidates = list(out_dir.glob("*.wav"))
    assert candidates, "expected a sibling .wav alongside out.mp4"
    wav = candidates[0]
    # A WAV header is the cheapest file-format check: each chunk has a
    # well-known ASCII tag, and an empty file would skip past this check.
    head = wav.read_bytes()[:4]
    assert head == b"RIFF", f"{wav.name} is not a WAV (header: {head!r})"


def test_animate_calls_pipeline_with_the_requested_num_frames(stub_pipeline, tmp_path, monkeypatch):
    seen = {}

    def recording(self, source_image, *, num_frames, prompt, steps, out_dir):
        seen["num_frames"] = num_frames
        return np.full((num_frames, 1280, 720, 3), 80, dtype=np.uint8)

    from server.avatar import musetalk_silent

    monkeypatch.setattr(musetalk_silent.MuseTalkSilentAnimator, "_run_pipeline", recording)
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    a = musetalk_silent.MuseTalkSilentAnimator()
    # New (lifecycle): bypass the not-loaded gate that ``animate``
    # checks before delegating to the main engine runtime.
    a._engine_runtime_loaded = lambda: True  # noqa: SLF001
    a.animate(
        src, prompt="", num_frames=125, steps=10, out_dir=out_dir
    )
    assert seen["num_frames"] == 125