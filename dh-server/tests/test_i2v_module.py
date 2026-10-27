"""CogVideoX-5B-I2V adapter.

The real model is heavyweight (~10 GB weights, GPU inference) and is *not*
available in CI, so these tests assert only the contract the pipeline relies
on: N frames at the project canvas, raised cleanly when weights are missing,
and a per-call injection point so the pipeline can swap in a stub.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (640, 800), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _save_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png())


def test_module_exposes_generate_class():
    # The pipeline imports one symbol by name; keep the surface small.
    from dh_server.avatar import i2v

    assert hasattr(i2v, "CogVideoXI2V")
    assert hasattr(i2v, "I2VUnavailable")


def test_missing_weights_raises_friendly_error(tmp_path):
    from dh_server.avatar.i2v import CogVideoXI2V, I2VUnavailable

    source = tmp_path / "src.png"
    _save_png(source)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Empty weights dir -> the constructor's own preflight must refuse, not
    # defer the failure to the first call.
    empty_weights = tmp_path / "none"
    empty_weights.mkdir()

    gen = CogVideoXI2V(weights_dir=empty_weights)
    with pytest.raises(I2VUnavailable) as exc:
        gen.generate(source, num_frames=49, steps=25, prompt="hi", seed=0, out_dir=out_dir)
    # Operator can find the fetch script from the message.
    assert "fetch_cogvidex_weights.sh" in str(exc.value)


def test_generate_returns_frames_at_canvas_size(tmp_path, monkeypatch):
    from dh_server.avatar import i2v

    # Inject a stub backend so we don't need torch / a GPU in CI.
    captured = {}

    def fake_backend(self, source_path, *, num_frames, steps, prompt, seed, device):
        captured["num_frames"] = num_frames
        captured["steps"] = steps
        captured["prompt"] = prompt
        captured["seed"] = seed
        # Return N frames at the 720x1280 canvas; the pipeline feeds these
        # straight into the per-frame MuseTalk prep loop.
        return [
            np.full((1280, 720, 3), fill_value=80, dtype=np.uint8) for _ in range(num_frames)
        ]

    monkeypatch.setattr(i2v.CogVideoXI2V, "_run_diffusion", fake_backend)

    source = tmp_path / "src.png"
    _save_png(source)
    weights = tmp_path / "weights" / "cogvideox-5b-i2v"
    weights.mkdir(parents=True)
    # The constructor's preflight just looks for *something* under the dir;
    # dropping a placeholder file is enough.
    (weights / ".gitkeep").write_bytes(b"")

    gen = i2v.CogVideoXI2V(weights_dir=weights)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    frames = gen.generate(
        source, num_frames=49, steps=25, prompt="look up", seed=42, out_dir=out_dir
    )
    assert len(frames) == 49
    for f in frames:
        assert f.shape == (1280, 720, 3)
        assert f.dtype == np.uint8

    assert captured["num_frames"] == 49
    assert captured["steps"] == 25
    assert captured["prompt"] == "look up"
    assert captured["seed"] == 42


def test_generate_writes_i2v_clip_artifact(tmp_path, monkeypatch):
    """The operator-visible artifact is the generated mp4 on disk.

    The pipeline then re-decodes it via ``video.decode_frames`` for the
    downstream prep loop, the same way it would a user-uploaded mp4.
    """
    from dh_server.avatar import i2v

    def fake_backend(self, source_path, *, num_frames, steps, prompt, seed, device):
        return [np.full((1280, 720, 3), 80, dtype=np.uint8) for _ in range(num_frames)]

    monkeypatch.setattr(i2v.CogVideoXI2V, "_run_diffusion", fake_backend)

    source = tmp_path / "src.png"
    _save_png(source)
    weights = tmp_path / "weights" / "cogvideox-5b-i2v"
    weights.mkdir(parents=True)
    (weights / ".gitkeep").write_bytes(b"")

    gen = i2v.CogVideoXI2V(weights_dir=weights)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gen.generate(source, num_frames=49, steps=25, prompt="x", seed=1, out_dir=out_dir)

    clip = out_dir / "i2v_clip.mp4"
    assert clip.exists()
    assert clip.stat().st_size > 0


def test_generate_requires_sentencepiece(tmp_path, monkeypatch):
    # The T5 tokenizer ships its model as spiece.model and needs the
    # SentencePiece native lib to load. Diffusers' fallback to TikToken
    # silently produces wrong tokens for T5's vocabulary, so the adapter
    # preflights the import and raises a clear instruction instead.
    from dh_server.avatar import i2v

    # The real backend is heavy (torch + diffusers) and unneeded here. Stub
    # the part of it that the preflight guards. We assert what an operator
    # sees: the I2VUnavailable carried up from the backend's first line.
    def fake_backend(self, source_path, *, num_frames, steps, prompt, seed, device):
        # Mirror the adapter's preflight in isolation so we don't depend
        # on the torch stack being importable in this test env.
        try:
            i2v.CogVideoXI2V._check_tokenizer_runtime_deps()
        except i2v.I2VUnavailable:
            raise
        return [np.full((1280, 720, 3), 80, dtype=np.uint8) for _ in range(num_frames)]

    monkeypatch.setattr(i2v.CogVideoXI2V, "_run_diffusion", fake_backend)

    # Make ``import sentencepiece`` raise ImportError without touching
    # the user's venv state. The preflight sits at the top of the
    # backend, so blocking ``import sentencepiece`` makes it fail loud.
    import builtins as _b
    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentencepiece" or name.startswith("sentencepiece."):
            raise ImportError("simulated: sentencepiece not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)

    source = tmp_path / "src.png"
    _save_png(source)
    weights = tmp_path / "weights" / "cogvideox-5b-i2v"
    weights.mkdir(parents=True)
    (weights / ".gitkeep").write_bytes(b"")

    gen = i2v.CogVideoXI2V(weights_dir=weights)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(i2v.I2VUnavailable) as exc:
        gen.generate(source, num_frames=49, steps=25, prompt="x", seed=1, out_dir=out_dir)
    msg = str(exc.value)
    assert "sentencepiece" in msg
    assert "pip install" in msg


def test_generate_requires_protobuf(tmp_path, monkeypatch):
    # sentencepiece alone is not enough -- the protobuf runtime is what
    # actually decodes spiece.model into tensor-friendly IDs. Same silent
    # TikToken fallback applies if it is missing: garbage video that
    # doesn't *look* broken. The preflight must catch this too.
    from dh_server.avatar import i2v

    def fake_backend(self, source_path, *, num_frames, steps, prompt, seed, device):
        i2v.CogVideoXI2V._check_tokenizer_runtime_deps()
        return [np.full((1280, 720, 3), 80, dtype=np.uint8) for _ in range(num_frames)]

    monkeypatch.setattr(i2v.CogVideoXI2V, "_run_diffusion", fake_backend)

    import builtins as _b
    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google.protobuf" or name == "protobuf" or name.startswith("google.protobuf."):
            raise ImportError("simulated: protobuf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)

    source = tmp_path / "src.png"
    _save_png(source)
    weights = tmp_path / "weights" / "cogvideox-5b-i2v"
    weights.mkdir(parents=True)
    (weights / ".gitkeep").write_bytes(b"")

    gen = i2v.CogVideoXI2V(weights_dir=weights)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(i2v.I2VUnavailable) as exc:
        gen.generate(source, num_frames=49, steps=25, prompt="x", seed=1, out_dir=out_dir)
    msg = str(exc.value)
    assert "protobuf" in msg
    assert "pip install" in msg


def test_tokenizer_preflight_lists_both_dependencies():
    # The preflight must mention every tokenizer dep, so an operator who
    # hits *one* missing install gets a single ``pip install`` line that
    # resolves everything in one go (rather than fixing one, retrying,
    # and discovering the next missing wheel).
    from dh_server.avatar import i2v

    deps = i2v.CogVideoXI2V._REQUIRED_TOKENIZER_DEPS
    # Each entry is ``(importable_module, pip_name)``; the preflight's
    # failure message uses the pip names.
    pips = [d[1] for d in deps]
    assert "sentencepiece" in pips
    assert "protobuf" in pips
