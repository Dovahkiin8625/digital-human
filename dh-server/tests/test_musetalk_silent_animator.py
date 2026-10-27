"""MuseTalk silent animator for the 形象活化 tab's "复用基础设施" slot.

MuseTalk is, by design, audio-driven: it retargets the source face's mouth /
jaw onto a driving audio waveform. With a synthesized silent WAV, the
output is *subtle* -- small jaw moves without phonemes -- which matches
"让一张静态人像轻微的动起来" exactly.

The animation harness reuses the project's existing muxer + encoder:
a tiny silent WAV synthesised in-process, the real MuseTalk pipeline
(``MuseTalkAvatarPipeline.process`` + ``MuseTalkEngine.drive_sync``) to
generate per-frame warped-face frames, and the same PyAV libx264 H.264
encoder the rest of the codebase uses to write ``out.mp4``.

Tests in this module use the ``stub_pipeline`` fixture to replace the
real MuseTalk synthesis with a constant frame stack so the torch-free
venv can exercise the wrapper (WAV synthesis, encode, mp4 existence)
without any GPU work. The ``test_animate_drives_*`` tests instead
exercise the real wiring with stubbed heavy bits (pipeline + engine).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from dh_server.avatar import animators


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Replace the real MuseTalk pipeline with a constant frame stack.

    ``_run_pipeline`` is the seam tests use to keep the torch-free venv
    from loading MuseTalk: production wires it to ``MuseTalkAvatarPipeline``
    + the engine's ``drive_sync`` (covered by the ``test_animate_drives_*``
    tests below), CI / unit tests stub it via this fixture so a single
    ``animate(...)`` call exercises the wrapper -- WAV synthesis, encode,
    mp4 existence -- without any GPU work.
    """
    from dh_server.avatar import musetalk_silent

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
    from dh_server.avatar import musetalk_silent  # noqa: F401 -- side effect

    assert "musetalk-silent" in animators.ANIMATORS
    assert animators.ANIMATORS["musetalk-silent"].name == "musetalk-silent"


def test_meta_groups_under_existing_infra_bucket():
    from dh_server.avatar import musetalk_silent

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
    from dh_server.avatar import musetalk_silent

    assert musetalk_silent.MuseTalkSilentAnimator().is_ready is True


# ----- validation -----


def test_validate_rejects_zero_or_negative_frames():
    from dh_server.avatar import musetalk_silent

    a = musetalk_silent.MuseTalkSilentAnimator()
    assert "num_frames" in a.validate(prompt="ignored", num_frames=0, steps=10)
    assert "num_frames" in a.validate(prompt="ignored", num_frames=-5, steps=10)


def test_validate_accepts_a_normal_motion_clip():
    from dh_server.avatar import musetalk_silent

    a = musetalk_silent.MuseTalkSilentAnimator()
    assert a.validate(prompt="ignored", num_frames=125, steps=10) == {}
    assert a.validate(prompt="ignored", num_frames=25, steps=10) == {}


def test_validate_rejects_too_long():
    from dh_server.avatar import musetalk_silent

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

    from dh_server.avatar import musetalk_silent

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


# ----- real path: _run_pipeline wires to MuseTalkAvatarPipeline + the engine -----
#
# These tests replace the placeholder default with the production intent:
# ``animate()`` builds a transient avatar pack, drives the singleton
# MuseTalk engine with the synthesised silent WAV, and trims the result
# to ``num_frames``. The mocks below stand in for the heavy bits so
# the tests run in the torch-free venv.


def _stub_avatar_pack_config():
    """The ``{"musetalk": {...}}`` shape ``MuseTalkAvatarPipeline.process``
    returns and ``_load_image_avatar`` reads. ``out_dir`` is filled by the
    production caller; ``tmp_path`` substitutes here.
    """
    return {
        "musetalk": {
            "mode": "image",
            "dir": "musetalk",
            "version": "1.5",
            "crop": 256,
            "detector": "mediapipe.face_landmarker",
            "frame_count": 1,
            "fps": 25,
            "source_fps": 25.0,
            "base_image": "normalized.png",
            "source": "src.png",
            "latents": "musetalk/latents.pt",
            "coords": "musetalk/coords.json",
            "masks": "musetalk/masks",
            "bbox": [10, 10, 100, 100],
            "mask_crop_box": [0, 0, 256, 256],
        }
    }


class _FakeEngine:
    """Records drive_sync invocations; returns the frames the test asks for."""

    def __init__(self, frames_to_return: list):
        self.calls: list[dict] = []
        self._frames = frames_to_return

    def drive_sync(self, pack, audio, sample_rate):
        self.calls.append(
            {
                "pack": pack,
                "audio": audio,
                "sample_rate": sample_rate,
            }
        )
        return self._frames


def _build_animator_with_stubs(monkeypatch, tmp_path, frames_to_return):
    """Stand up an animator with the heavy deps replaced; return animator + engine.

    Writes the dummy latents/normalized.png the loader expects so any code path
    that touches disk for the real AvatarPack can run without IO errors. Tests
    that don't need those files can ignore them.
    """
    from dh_server.avatar import musetalk_silent
    from dh_server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline

    # Stub the avatar pipeline: process() returns a stub config without
    # running VAE/mediapipe/face parsing (those need torch + a real image).
    monkeypatch.setattr(
        MuseTalkAvatarPipeline,
        "process",
        lambda self, avatar_id, source_path, out_dir, progress: _stub_avatar_pack_config(),
    )

    engine = _FakeEngine(frames_to_return=frames_to_return)
    # Patch the module-level binding inside musetalk_silent. The import is
    # hoisted (``from dh_server.main import get_session_manager``) so this
    # is the binding production code resolves. The wrapped object only
    # needs ``engine``; other attributes are unused by the i2v path.
    monkeypatch.setattr(
        musetalk_silent,
        "get_session_manager",
        lambda: SimpleNamespace(engine=engine),
    )

    a = musetalk_silent.MuseTalkSilentAnimator()
    a._engine_runtime_loaded = lambda: True  # noqa: SLF001
    return a, engine


def test_animate_drives_engine_with_silent_audio(monkeypatch, tmp_path):
    """The production default must invoke engine.drive_sync with the silent WAV.

    The placeholder _run_pipeline never touches the engine, so this test
    fails on the un-wired default. After the fix, animate(num_frames=125)
    must call engine.drive_sync exactly once with audio matching the
    synthesised 16 kHz silent WAV (5 s = 80000 samples) and the freshly
    built avatar pack.
    """
    frames = [
        _frame(j, (720, 1280, 3)) for j in range(125)
    ]
    a, engine = _build_animator_with_stubs(monkeypatch, tmp_path, frames_to_return=frames)

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    a.animate(src, prompt="", num_frames=125, steps=10, out_dir=out_dir)

    assert len(engine.calls) == 1, f"expected 1 drive_sync call, got {len(engine.calls)}"
    call = engine.calls[0]
    # 5 s of 16 kHz mono int16 = 80000 samples -> float32 in [-1, 1].
    assert call["audio"].shape == (80000,), f"audio shape: {call['audio'].shape}"
    assert call["audio"].dtype == np.float32
    assert call["sample_rate"] == 16000
    assert call["pack"].avatar_id == "i2v-silent"
    # Out.mp4 must exist; the encode runs after _run_pipeline returns.
    assert (out_dir / "out.mp4").exists()
    assert (out_dir / "out.mp4").stat().st_size > 0


def test_animate_trims_to_num_frames(monkeypatch, tmp_path):
    """_write_silent_wav uses max(0.5, num_frames/25.0)s, so the engine
    emits slightly more frames than the operator requested for small
    num_frames (e.g. num_frames=1 -> 12 frames from 0.5 s audio). The
    production default must trim to exactly num_frames so the operator
    gets the clip length they asked for.

    The placeholder returns num_frames copies of the source image, so
    this test fails on the un-wired default with the wrong pixel values.
    """
    # Twelve distinct frames: each frame i has a unique pixel value so we
    # can detect both the trim (shape == (1, ...)) and that the kept frame
    # is the FIRST engine output, not any of the others.
    frames = [_frame(j, (720, 1280, 3), fill=j + 1) for j in range(12)]
    a, engine = _build_animator_with_stubs(monkeypatch, tmp_path, frames_to_return=frames)

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    a.animate(src, prompt="", num_frames=1, steps=10, out_dir=out_dir)

    # The encode path inspects the rgb stack via the shared helper; spy
    # on it to read what animate() actually handed to the encoder.
    # _encode_h264 is imported locally in animate() from dh_server.avatar.video,
    # so the patch must live on the source module.
    from dh_server.avatar import video as video_mod

    captured: dict = {}
    real_encode = video_mod._encode_h264

    def spy(path, rgb, fps=25):
        captured["rgb"] = np.asarray(rgb)
        captured["fps"] = fps
        return real_encode(path, rgb, fps=fps)

    monkeypatch.setattr("dh_server.avatar.video._encode_h264", spy)

    # Re-run so the spy captures the encode (the first run already wrote
    # out.mp4 via the un-spied encode; we want to read the rgb stack).
    out_dir2 = tmp_path / "out2"
    a.animate(src, prompt="", num_frames=1, steps=10, out_dir=out_dir2)
    assert "rgb" in captured, "encode_h264 was not called"

    rgb = captured["rgb"]
    assert rgb.shape[0] == 1, f"expected 1 frame after trim, got shape {rgb.shape}"
    # The kept frame is frame 0 of the engine's output (fill=1), not the
    # placeholder's copy of the source image (fill=(10, 20, 30)). The PIL
    # round-trip in animate() may permute H/W; check the unique pixel value
    # survived regardless of axis order.
    unique = {tuple(p) for p in rgb.reshape(-1, 3).tolist()}
    assert unique == {(1, 1, 1)}, f"expected frame 0's fill value, got {unique}"


def test_animate_wraps_pipeline_face_error(monkeypatch, tmp_path):
    """``MuseTalkAvatarPipeline.process`` raises ``PipelineError`` when
    the image has no face; ``animate()`` must surface that as
    ``AnimatorError`` so the i2v HTTP layer records it on the task row.

    The placeholder never calls process(), so no exception is raised;
    the test fails on the un-wired default.
    """
    from dh_server.avatar import animators
    from dh_server.avatar import musetalk_silent
    from dh_server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline
    from dh_server.avatar.pipeline import PipelineError

    def raise_no_face(self, avatar_id, source_path, out_dir, progress):
        raise PipelineError("no face detected in the uploaded image")

    monkeypatch.setattr(MuseTalkAvatarPipeline, "process", raise_no_face)
    monkeypatch.setattr(
        musetalk_silent,
        "get_session_manager",
        lambda: SimpleNamespace(engine=_FakeEngine(frames_to_return=[])),
    )
    a = musetalk_silent.MuseTalkSilentAnimator()
    a._engine_runtime_loaded = lambda: True  # noqa: SLF001

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"

    with pytest.raises(animators.AnimatorError) as excinfo:
        a.animate(src, prompt="", num_frames=125, steps=10, out_dir=out_dir)

    # The animate() wrapper composes ``MuseTalk silent inference failed: <inner>``.
    assert "MuseTalk silent" in str(excinfo.value)
    assert "no face" in str(excinfo.value)


def _frame(j: int, shape: tuple, *, fill: int | None = None) -> "_FrameLike":
    """A tiny ``DriveFrame``-shaped stand-in the FakeEngine can return."""
    from dh_server.engine.base import DriveFrame

    value = fill if fill is not None else (j + 1)
    return DriveFrame(pts=j / 25.0, image=np.full(shape, value, dtype=np.uint8))