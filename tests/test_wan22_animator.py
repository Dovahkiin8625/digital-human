"""Wan 2.2 I2V portrait animator for the 形象活化 tab.

A single 5 B dense checkpoint (``Wan-AI/Wan2.2-TI2V-5B-Diffusers``) drives
the OSS I2V backend, registered as ``wan22-i2v-5b-bf16`` (~22 GB VRAM).
The diffusers stack is heavy (a 5 B model loads ~15 s the first time), so
the tests monkey-patch ``_run_pipeline`` and assert the contract instead.

An fp8 sibling was originally planned but not shipped: Wan-AI hasn't
released an officially-quantised fp8 release of TI2V-5B, and the third-
party quants (Comfy-Org / bullerwins) don't fit diffusers'
``from_pretrained``. Re-evaluate when Wan-AI ships an fp8 release.

``is_ready`` returns False on hosts that can't satisfy the variant's VRAM
budget; the page renders whatever the registry exposes and surfaces any
``AnimatorError`` inline if the operator picks something the host can't
actually run.
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
    """Inject a deterministic 720x1280 frame stack in place of diffusers."""
    from server.avatar import wan22

    def fake(self, source_image, *, prompt, num_frames, steps, out_dir):
        # Save the prompt + knob values into a sidecar so tests can read back
        # what the animator actually sent to the pipeline.
        np.save(
            out_dir / "_frames.npy",
            np.full((num_frames, 1280, 720, 3), 80, dtype=np.uint8),
        )
        (out_dir / "_call.txt").write_text(
            f"prompt={prompt};num_frames={num_frames};steps={steps}"
        )
        return np.full((num_frames, 1280, 720, 3), 80, dtype=np.uint8)

    monkeypatch.setattr(wan22.Wan22I2VAnimator, "_run_pipeline", fake)


def _seed_weights(root, *, variant_name):
    """Drop a placeholder file where the animator looks for the snapshot."""
    from server.avatar import wan22

    spec = wan22.Wan22I2VAnimator.VARIANTS[variant_name]
    d = root / "models" / spec["weights_subdir"]
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitkeep").write_bytes(b"")


def _gpu_vram_gb(monkeypatch, gb: float):
    """Force ``torch.cuda.get_device_properties().total_memory`` to read N GB."""
    import torch

    class _FakeProps:
        def __init__(self, gb):
            self.total_memory = int(gb * 1024 ** 3)

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _idx: _FakeProps(gb)
    )


# ----- registry / meta -----


def test_variant_registers_on_import():
    # The bf16 variant must be in the registry on import. A future variant
    # (e.g. fp8 once Wan-AI ships one) gets its line in this test when it
    # lands.
    from server.avatar import wan22  # noqa: F401 -- side effect: registers the variant

    expected = {"wan22-i2v-5b-bf16"}
    registered = set(animators.ANIMATORS)
    for name in expected:
        assert name in registered, f"variant {name} not registered"


def test_meta_lists_ui_keys():
    from server.avatar import wan22

    for animator in wan22.all_animators():
        meta = animator.meta()
        for key in (
            "label_zh",
            "group_zh",
            "description",
            "vram_hint",
            "accepts_prompt",
            "accepts_num_frames",
            "accepts_steps",
            "num_frames_min",
            "num_frames_max",
        ):
            assert key in meta, f"{animator.name}.meta() missing {key!r}"
        # Wan accepts all three knobs -- general-purpose i2v.
        assert meta["accepts_prompt"] is True
        assert meta["accepts_num_frames"] is True
        assert meta["accepts_steps"] is True


def test_meta_groups_under_oss_i2v():
    from server.avatar import wan22

    for animator in wan22.all_animators():
        group = animator.meta()["group_zh"]
        assert group == "开源参考生视频", (
            f"variant {animator.name} must group under the OSS dropdown bucket"
        )


def test_num_frames_bounds_land_on_grid():
    from server.avatar import wan22

    for animator in wan22.all_animators():
        m = animator.meta()
        # Wan's num_frames must be 16N+1; the form's range mirrors that.
        assert (m["num_frames_max"] - 1) % 16 == 0, (
            f"{animator.name}: max must be 16N+1 to land on Wan's grid"
        )
        assert (m["num_frames_min"] - 1) % 16 == 0, (
            f"{animator.name}: min must be 16N+1"
        )
        # Accepts at least the smallest grid cell (17 frames) and not less;
        # upper bound is enough for a ~5 s clip @ 24 fps.
        assert m["num_frames_min"] >= 17
        assert m["num_frames_max"] >= 81


def test_meta_pins_bf16_dtype():
    # A typo here would silently fall back to fp32 and OOM the consumer
    # card, so the spec pins the dtype explicitly.
    from server.avatar import wan22

    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=None)
    assert a.spec["torch_dtype"] == "bfloat16"


# ----- readiness -----


def test_is_ready_false_when_weights_missing(tmp_path, monkeypatch):
    from server.avatar import wan22

    monkeypatch.setattr(wan22, "WAN22_WEIGHTS_ROOT", tmp_path / "no-models", raising=False)
    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path / "no-models")
    assert a.is_ready is False


def test_is_ready_true_when_weights_and_gpu_match(tmp_path, monkeypatch):
    from server.avatar import wan22

    # The production layout is ``data_dir/models/<Wan2.2-...>``; mirror it
    # in tests by seeding under ``tmp_path/models``. 16 GB is the consumer
    # baseline -- with sequential_cpu_offload the model runs at 10 GB peak.
    _seed_weights(tmp_path, variant_name="5b-bf16")
    _gpu_vram_gb(monkeypatch, 16.0)
    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path / "models")
    assert a.is_ready is True


def test_is_ready_false_when_gpu_below_budget(tmp_path, monkeypatch):
    from server.avatar import wan22

    _seed_weights(tmp_path, variant_name="5b-bf16")
    _gpu_vram_gb(monkeypatch, 8.0)
    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path / "models")
    assert a.is_ready is False, (
        "5b-bf16 needs ~12 GB (10 GB peak + 2 GB headroom); "
        "an 8 GB host should refuse to claim readiness"
    )


# ----- validation -----


def test_validate_rejects_num_frames_off_grid():
    from server.avatar import wan22

    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=None)
    # 50 is not 16N+1; 49 = 16*3+1 IS on the grid (sanity check below).
    errs = a.validate(prompt="x", num_frames=50, steps=25)
    assert "num_frames" in errs
    # Sanity: 49 lands on the grid and is accepted.
    assert "num_frames" not in a.validate(prompt="x", num_frames=49, steps=25)


def test_validate_accepts_num_frames_on_grid():
    from server.avatar import wan22

    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=None)
    # 81 = 16*5 + 1; 17 = 16*1 + 1; 113 = 16*7 + 1 (the upper bound).
    assert a.validate(prompt="x", num_frames=81, steps=25) == {}
    assert a.validate(prompt="x", num_frames=17, steps=25) == {}
    assert a.validate(prompt="x", num_frames=113, steps=25) == {}


def test_validate_rejects_steps_out_of_range():
    from server.avatar import wan22

    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=None)
    assert "steps" in a.validate(prompt="x", num_frames=81, steps=1)
    assert "steps" in a.validate(prompt="x", num_frames=81, steps=999)


# ----- animate -----


def test_animate_writes_mp4_and_forwards_knobs(stub_pipeline, tmp_path, monkeypatch):
    from server.avatar import wan22

    _seed_weights(tmp_path, variant_name="5b-bf16")
    _gpu_vram_gb(monkeypatch, 16.0)
    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path / "models")

    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    out_dir = tmp_path / "out"
    a.animate(src, prompt="look up", num_frames=81, steps=20, out_dir=out_dir)

    assert (out_dir / "out.mp4").exists()
    assert (out_dir / "out.mp4").stat().st_size > 0
    # The exact prompt + knobs reached the pipeline. If the adapter dropped
    # or mangled them the call file would say so.
    call = (out_dir / "_call.txt").read_text()
    assert "prompt=look up" in call
    assert "num_frames=81" in call
    assert "steps=20" in call


def test_animate_raises_when_weights_missing(tmp_path, monkeypatch):
    from server.avatar import wan22

    monkeypatch.setattr(wan22, "WAN22_WEIGHTS_ROOT", tmp_path / "none", raising=False)
    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path / "none")
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    with pytest.raises(animators.AnimatorError) as exc:
        a.animate(src, prompt="", num_frames=81, steps=20, out_dir=tmp_path / "out")
    assert "weights" in str(exc.value).lower() or "wan22" in str(exc.value).lower()


def test_animate_raises_when_validation_fails(tmp_path):
    from server.avatar import wan22

    a = wan22.Wan22I2VAnimator(variant="5b-bf16", weights_root=tmp_path)
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    # Off-grid num_frames is rejected before any weights read or VRAM probe.
    with pytest.raises(animators.AnimatorError) as exc:
        a.animate(src, prompt="x", num_frames=50, steps=20, out_dir=tmp_path / "out")
    assert "num_frames" in str(exc.value)
