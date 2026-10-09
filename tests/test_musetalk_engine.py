"""Contract tests for the real MuseTalk engine, with no GPU and no torch.

The engine's model layer is stubbed at the `_Runtime` boundary, so what these
pin is the part that is easy to get wrong and expensive to debug on a GPU:
the frame accumulator, the 48->16 kHz resampler, the rolling window's frame
alignment, the offload, and the pack contract.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from server.engine import musetalk_engine as me
from server.engine.base import AvatarPack, EngineUnavailable, create_engine
from server.engine.musetalk_engine import MuseTalkEngine, resample_to_16k


def _pack(tmp_path, prep=None):
    img = tmp_path / "normalized.png"
    Image.new("RGB", (720, 1280), (5, 5, 5)).save(img)
    return AvatarPack(
        avatar_id="a", root=tmp_path, source_image_path=img, config={}, prep=prep or {}
    )


def _prep(tmp_path):
    """A pack.json-shaped musetalk block plus the sidecars it points at."""
    sidecar = tmp_path / "musetalk"
    sidecar.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), (7, 7, 7)).save(sidecar / "face_crop.png")
    Image.new("L", (272, 272), 255).save(sidecar / "mask.png")
    return {
        "dir": "musetalk",
        "bbox": [100, 200, 300, 400],
        "mask_crop_box": [50, 150, 350, 450],
        "base_image": "normalized.png",
        "latents": "musetalk/latents.pt",
        "mask": "musetalk/mask.png",
    }


class _StubRuntime:
    """Stands in for the loaded VAE/UNet/whisper triple."""

    device = "cpu"
    weight_dtype = None
    whisper = None

    def __init__(self, prompts_per_call=8):
        self.prompts_per_call = prompts_per_call
        self.calls = 0
        self.last_window = None

    class _FeatureExtractor:
        def __call__(self, window, return_tensors=None, sampling_rate=None):
            class _Out:
                input_features = np.zeros((1, 80, 3000), dtype=np.float32)

            return _Out()

    class _AudioProcessor:
        def get_whisper_chunk(self, feats, device, dtype, whisper, length, fps=25):
            # Mirrors MuseTalk: one prompt per output frame, 50-wide.
            n = max(1, int(np.floor(length / 16000 * fps)))
            # Each frame's prompt is tagged with its index so a test can tell
            # which window-relative frame the engine actually selected.
            arr = np.zeros((n, 50, 384), dtype=np.float32)
            arr[:, 0, 0] = np.arange(n)
            return _FakePrompts(arr)

    feature_extractor = _FeatureExtractor()
    audio_processor = _AudioProcessor()

    def pe(self, wb):
        return wb


class _FakePrompts:
    """Minimal stand-in for a torch tensor: shape, list indexing, .to()."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    @property
    def shape(self):
        return self.arr.shape

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, item):
        return _FakePrompts(self.arr[item])

    def to(self, *args, **kwargs):
        return self


class _Moved:
    """Records where `_load_latents` was pointed and ignores `.to()`."""

    def __init__(self, path):
        self.path = path

    def to(self, *args, **kwargs):
        return self


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    monkeypatch.setattr(eng, "_ensure_runtime", lambda: _StubRuntime())
    monkeypatch.setattr(
        eng, "_render", lambda rt, avatar, prompts: [np.zeros((1280, 720, 3), np.uint8)] * len(prompts.arr)
    )
    monkeypatch.setattr(
        eng,
        "_load_avatar",
        lambda rt, pack: me._AvatarState(
            bbox=[100, 200, 300, 400],
            base_bgr=np.zeros((1280, 720, 3), np.uint8),
            latents=None,
            mask=np.zeros((272, 272), np.uint8),
            mask_crop_box=[50, 150, 350, 450],
        ),
    )
    return eng


def _drive(eng, pack, ctx, n_blocks, block_ms=100, rate=48000):
    frames = []
    samples = rate * block_ms // 1000
    for _ in range(n_blocks):
        frames += asyncio.run(
            eng.drive(pack, np.zeros(samples, dtype=np.float32), rate, context=ctx)
        )
    return frames


# -- resampler ----------------------------------------------------------

def test_resample_48k_to_16k_length_and_dtype():
    x = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)
    y = resample_to_16k(x, 48000)
    assert y.dtype == np.float32
    # Exactly N/ratio: a longer output would gain ~2 samples per block and the
    # 25 fps frame grid would drift against the ingest timeline.
    assert len(y) == 16000


def test_resample_has_no_group_delay():
    """An impulse must land at exactly input_index/3.

    If the decimator carried the filter's ~16-sample group delay, every block
    would be shifted and the frame grid would slide.
    """
    x = np.zeros(4800, dtype=np.float32)
    x[1200] = 1.0
    y = resample_to_16k(x, 48000)
    assert int(np.argmax(y)) == 400


def test_resample_blocks_concatenate_without_a_seam():
    """Three consecutive 100 ms blocks must form one continuous timeline."""
    t = np.arange(9600) / 48000.0
    s = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    joined = np.concatenate([resample_to_16k(s[i : i + 4800], 48000) for i in (0, 4800)])
    assert len(joined) == 3200
    ref = np.sin(2 * np.pi * 300 * np.arange(3200) / 16000.0)
    # The only deviation is the FIR's ~5-sample edge transient at each block
    # boundary; a real group-delay error would show as a whole-block phase
    # shift and blow this up to ~0.12.
    assert float(np.abs(joined[50:3150] - ref[50:3150]).max()) < 0.02
    assert float(np.abs(joined[100:1500] - ref[100:1500]).max()) < 1e-3


def test_resample_preserves_dc_gain():
    assert float(resample_to_16k(np.ones(4800, np.float32), 48000)[100]) == pytest.approx(1.0)


def test_resample_passthrough_at_16k():
    x = np.ones(1600, dtype=np.float32)
    assert resample_to_16k(x, 16000) is x


def test_resample_non_integer_ratio():
    y = resample_to_16k(np.ones(44100, dtype=np.float32), 44100)
    assert abs(len(y) - 16000) <= 1


# -- accumulator --------------------------------------------------------

def test_ten_blocks_yield_exactly_25_frames(engine, tmp_path):
    """The whole point of the accumulator: 100 ms * 25 fps = 2.5, not 2.

    A per-block round() (banker's rounding) would emit 2 frames every block and
    silently run the stream at 20 fps.
    """
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    frames = _drive(engine, pack, ctx, 10)
    assert len(frames) == 25
    assert [f.image.shape for f in frames[:1]] == [(1280, 720, 3)]


def test_accumulator_emits_two_then_three(engine, tmp_path):
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    counts = []
    for _ in range(4):
        counts.append(
            len(asyncio.run(engine.drive(pack, np.zeros(4800, np.float32), 48000, context=ctx)))
        )
    assert counts == [2, 3, 2, 3]


def test_frame_pts_is_absolute(engine, tmp_path):
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    frames = _drive(engine, pack, ctx, 4)
    assert frames[0].pts == 0.0
    assert frames[1].pts == pytest.approx(1 / 25)
    assert frames[-1].pts == pytest.approx((len(frames) - 1) / 25)


def test_block_shorter_than_a_frame_emits_nothing(engine, tmp_path):
    """A 20 ms tail (0.5 frame) must not round up to a frame every call."""
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    assert asyncio.run(engine.drive(pack, np.zeros(960, np.float32), 48000, context=ctx)) == []


def test_context_is_per_session(engine, tmp_path):
    pack = _pack(tmp_path)
    a, b = engine.new_context(pack), engine.new_context(pack)
    _drive(engine, pack, a, 4)
    assert b.emitted == 0
    assert a.emitted == 10


# -- rolling whisper window --------------------------------------------

def test_prompt_index_clamps_to_available_frames(engine, tmp_path):
    """Asking for more frames than whisper returned must not IndexError."""
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    ctx.buf16k = np.zeros(640, dtype=np.float32)
    prompts = engine._prompts_for(engine._ensure_runtime(), ctx, 0, 20)
    assert prompts.shape[0] == 20


def test_window_anchors_on_a_frame_boundary(engine, tmp_path, monkeypatch):
    """Window start must be an exact multiple of 640 samples (one 25 fps frame)."""
    seen = {}

    class _Rec(_StubRuntime):
        class _AudioProcessor(_StubRuntime._AudioProcessor):
            def get_whisper_chunk(self, feats, device, dtype, whisper, length, fps=25):
                seen["length"] = length
                return super().get_whisper_chunk(feats, device, dtype, whisper, length, fps)

        audio_processor = _AudioProcessor()

    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    ctx.buf16k = np.zeros(16000, dtype=np.float32)  # 1 s
    engine._prompts_for(_Rec(), ctx, 20, 23)  # first=20 -> f0=15 -> start=9600
    assert seen["length"] == 16000 - 15 * 640


def test_prompt_selection_maps_to_absolute_frames(engine, tmp_path):
    """Frame f must get the prompt tagged f - f0, not a window-relative index."""
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    ctx.buf16k = np.zeros(16000, dtype=np.float32)
    rt = engine._ensure_runtime()

    prompts = engine._prompts_for(rt, ctx, 20, 23)  # f0 = 15
    # The stub tags prompt[i] with i, so the selected prompts are 5,6,7.
    assert prompts.arr[:, 0, 0].astype(int).tolist() == [5, 6, 7]


def test_rolling_buffer_is_bounded(engine, tmp_path):
    pack = _pack(tmp_path)
    ctx = engine.new_context(pack)
    _drive(engine, pack, ctx, 40)  # 4 s of audio
    assert len(ctx.buf16k) <= me.MAX_BUF


# -- pack contract ------------------------------------------------------

def test_missing_pack_is_rejected(tmp_path):
    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    with pytest.raises(EngineUnavailable, match="no musetalk pack"):
        eng._load_avatar(_StubRuntime(), _pack(tmp_path, prep={}))


def test_load_avatar_reads_sidecars(tmp_path, monkeypatch):
    prep = _prep(tmp_path)
    # The .pt itself is torch's business; what this pins is which sidecar each
    # key names and how the coordinates reach the state object.
    with open(tmp_path / prep["latents"], "wb") as fh:
        np.save(fh, np.zeros((1, 8, 32, 32), np.float32))
    monkeypatch.setattr(me, "_load_latents", lambda path: _Moved(path))
    eng = MuseTalkEngine(models_dir=tmp_path / "m")

    state = eng._load_avatar(_StubRuntime(), _pack(tmp_path, prep=prep))

    assert state.bbox == [100, 200, 300, 400]
    assert state.mask_crop_box == [50, 150, 350, 450]
    assert state.base_bgr.shape == (1280, 720, 3)
    assert state.mask.shape == (272, 272)


def test_prepare_raises_without_weights(tmp_path):
    engine = MuseTalkEngine(models_dir=tmp_path / "missing")
    with pytest.raises(EngineUnavailable):
        asyncio.run(engine.prepare(_pack(tmp_path)))


def test_describe_never_loads_weights(tmp_path):
    engine = MuseTalkEngine(models_dir=tmp_path / "m")
    d = engine.describe()
    assert d["name"] == "musetalk"
    assert d["weights_present"] is False
    assert d["loaded"] is False
    assert isinstance(d["gpu"], bool)


def test_create_engine_musetalk():
    engine = create_engine("musetalk")
    assert isinstance(engine, MuseTalkEngine)
    assert engine.name == "musetalk"


# -- offload ------------------------------------------------------------

@pytest.mark.asyncio
async def test_drive_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """A blocking drive() must not stall the loop.

    The stub sleeps like a CUDA call would; a concurrent ticker has to keep
    ticking. Without the worker pool the loop would be frozen for the whole
    sleep and the ticker would show one huge gap.
    """
    import time

    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    monkeypatch.setattr(eng, "_ensure_runtime", lambda: _StubRuntime())
    monkeypatch.setattr(
        eng,
        "_load_avatar",
        lambda rt, pack: me._AvatarState(
            bbox=[100, 200, 300, 400],
            base_bgr=np.zeros((1280, 720, 3), np.uint8),
            latents=None,
            mask=np.zeros((272, 272), np.uint8),
            mask_crop_box=[50, 150, 350, 450],
        ),
    )

    def blocking_drive(pack, audio, sample_rate, ctx):
        time.sleep(0.3)
        return []

    monkeypatch.setattr(eng, "_drive_sync", blocking_drive)

    ticks = []
    stop = False

    async def ticker():
        while not stop:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.002)

    task = asyncio.create_task(ticker())
    t0 = time.monotonic()
    await eng.drive(
        _pack(tmp_path), np.zeros(4800, np.float32), 48000, context=eng.new_context(_pack(tmp_path))
    )
    elapsed = time.monotonic() - t0
    stop = True
    await task

    assert elapsed >= 0.3, "stub did not actually block"
    # Windows' timer granularity is ~15 ms, so count gaps rather than ticks: a
    # drive() awaited on the loop would leave one gap as long as the whole
    # blocking call, while a worker-pool offload leaves every gap at the tick
    # period.
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) >= 3, f"ticker never ran during the blocking call ({len(ticks)} ticks)"
    assert max(gaps) < 0.1, f"event loop was blocked for {max(gaps):.3f}s"


@pytest.mark.asyncio
async def test_closed_engine_refuses_work(tmp_path):
    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    await eng.close()
    with pytest.raises(EngineUnavailable, match="closed"):
        await eng.drive(_pack(tmp_path), np.zeros(4800, np.float32), 48000)


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path):
    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    await eng.close()
    await eng.close()  # must not raise on a second shutdown
