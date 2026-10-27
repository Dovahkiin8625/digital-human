"""Wan 2.1 I2V portrait animator for the 形象活化 tab.

Four variants ship side-by-side; each gets its own ``Wan21I2VAnimator`` instance
and its own ``ANIMATORS`` entry. The diffusers stack is heavy (a 14 B model
loads ~30 s the first time; ~5 s for 1.3 B) so the tests monkey-patch
``_run_pipeline`` and assert the contract instead.

Variants in order of VRAM cost:

* ``wan21-i2v-1_3b``        -- ~5 GB; fits any CUDA host
* ``wan21-i2v-14b-480p-int8`` -- ~14 GB; the one that fits RTX 5070 ti cleanly
* ``wan21-i2v-14b-480p``    -- ~28 GB; full-precision 480P
* ``wan21-i2v-14b-720p``    -- ~32 GB; full-precision 720P

``is_ready`` returns False on hosts that can't satisfy a variant's VRAM
budget; the page renders whatever the registry exposes and surface any
``AnimatorError`` inline if the user picks the wrong one.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dh_server.avatar import animators


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Inject a deterministic 720x1280 frame stack in place of diffusers."""
    from dh_server.avatar import wan21

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

    monkeypatch.setattr(wan21.Wan21I2VAnimator, "_run_pipeline", fake)


def _seed_weights(root, *, variant_name):
    """Drop a placeholder file where the animator looks for the snapshot."""
    from dh_server.avatar import wan21

    spec = wan21.Wan21I2VAnimator.VARIANTS[variant_name]
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


def test_all_four_variants_register_on_import():
    # The four variants must all be in the registry on import. A future
    # variant gets its line in this test when it lands.
    from dh_server.avatar import wan21  # noqa: F401 -- side effect: registers all four

    expected = {
        "wan21-i2v-1_3b",
        "wan21-i2v-14b-480p-int8",
        "wan21-i2v-14b-480p",
        "wan21-i2v-14b-720p",
    }
    registered = set(animators.ANIMATORS)
    for name in expected:
        assert name in registered, f"variant {name} not registered"


def test_each_variant_meta_lists_ui_keys():
    from dh_server.avatar import wan21

    for animator in wan21.all_animators():
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


def test_each_variant_meta_group_is_oss_i2v():
    from dh_server.avatar import wan21

    for animator in wan21.all_animators():
        group = animator.meta()["group_zh"]
        assert group == "开源参考生视频", (
            f"variant {animator.name} must group under the OSS dropdown bucket"
        )


def test_each_variant_num_frames_bounds_scale_with_resolution():
    from dh_server.avatar import wan21

    for animator in wan21.all_animators():
        m = animator.meta()
        # Wan's num_frames must be 16N+1; the form's range mirrors that.
        assert (m["num_frames_max"] - 1) % 16 == 0, (
            f"{animator.name}: max must be 16N+1 to land on Wan's grid"
        )
        assert (m["num_frames_min"] - 1) % 16 == 0, (
            f"{animator.name}: min must be 16N+1"
        )
        # All Wan variants accept at least the smallest grid cell (17 frames)
        # and not less; the upper bound is enough for a 5 s clip @ 16 fps.
        assert m["num_frames_min"] >= 17
        assert m["num_frames_max"] >= 81


def test_each_variant_vram_hint_distinct():
    from dh_server.avatar import wan21

    hints = {a.meta()["vram_hint"] for a in wan21.all_animators()}
    # We want operators to be able to tell two variants apart on the page.
    assert len(hints) == len(wan21.all_animators()), (
        "duplicate vram_hint means the dropdown labels collide"
    )


# ----- readiness -----


def test_is_ready_false_when_weights_missing(tmp_path, monkeypatch):
    from dh_server.avatar import wan21

    monkeypatch.setattr(wan21, "WAN21_WEIGHTS_ROOT", tmp_path / "no-models", raising=False)
    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=tmp_path / "no-models")
    assert a.is_ready is False


def test_is_ready_true_when_weights_and_gpu_match(tmp_path, monkeypatch):
    from dh_server.avatar import wan21

    # The production layout is ``data_dir/models/<Wan2.1-...>``; mirror it
    # in tests by seeding under ``tmp_path/models``.
    _seed_weights(tmp_path, variant_name="1_3b")
    _gpu_vram_gb(monkeypatch, 8.0)
    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=tmp_path / "models")
    assert a.is_ready is True


def test_is_ready_false_when_gpu_below_budget(tmp_path, monkeypatch):
    from dh_server.avatar import wan21

    _seed_weights(tmp_path, variant_name="14b-720p")
    _gpu_vram_gb(monkeypatch, 16.0)
    a = wan21.Wan21I2VAnimator(variant="14b-720p", weights_root=tmp_path / "models")
    assert a.is_ready is False, (
        "14b-720p needs ~32 GB; a 16 GB host should refuse to claim readiness"
    )


# ----- validation -----


def test_validate_rejects_num_frames_off_grid():
    from dh_server.avatar import wan21

    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=None)
    # 50 is not 16N+1; 49 = 16*3+1 IS on the grid (sanity check below).
    errs = a.validate(prompt="x", num_frames=50, steps=25)
    assert "num_frames" in errs
    # Sanity: 49 lands on the grid and is accepted.
    assert "num_frames" not in a.validate(prompt="x", num_frames=49, steps=25)


def test_validate_accepts_num_frames_on_grid():
    from dh_server.avatar import wan21

    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=None)
    # 81 = 16*5 + 1; 17 = 16*1 + 1.
    assert a.validate(prompt="x", num_frames=81, steps=25) == {}
    assert a.validate(prompt="x", num_frames=17, steps=25) == {}


def test_validate_rejects_steps_out_of_range():
    from dh_server.avatar import wan21

    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=None)
    assert "steps" in a.validate(prompt="x", num_frames=81, steps=1)
    assert "steps" in a.validate(prompt="x", num_frames=81, steps=999)


# ----- animate -----


def test_animate_writes_mp4_and_forwards_knobs(stub_pipeline, tmp_path, monkeypatch):
    from dh_server.avatar import wan21

    _seed_weights(tmp_path, variant_name="1_3b")
    _gpu_vram_gb(monkeypatch, 8.0)
    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=tmp_path / "models")

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
    from dh_server.avatar import wan21

    monkeypatch.setattr(wan21, "WAN21_WEIGHTS_ROOT", tmp_path / "none", raising=False)
    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=tmp_path / "none")
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    with pytest.raises(animators.AnimatorError) as exc:
        a.animate(src, prompt="", num_frames=81, steps=20, out_dir=tmp_path / "out")
    assert "weights" in str(exc.value).lower() or "wan21" in str(exc.value).lower()


def test_animate_raises_when_validation_fails(tmp_path):
    from dh_server.avatar import wan21

    a = wan21.Wan21I2VAnimator(variant="1_3b", weights_root=tmp_path)
    src = tmp_path / "src.png"
    Image.new("RGB", (640, 800), (10, 20, 30)).save(src)
    # Off-grid num_frames is rejected before any weights read or VRAM probe.
    with pytest.raises(animators.AnimatorError) as exc:
        a.animate(src, prompt="x", num_frames=50, steps=20, out_dir=tmp_path / "out")
    assert "num_frames" in str(exc.value)
