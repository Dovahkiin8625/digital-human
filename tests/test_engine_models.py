"""Engine describe() contracts for the Models admin tab + MuseTalk unload().

The admin System page exposes the live drive engine's runtime state so the
operator can see what's on the GPU and (for the real engine) free it. Both
describes must include a ``models`` list with one slot each; the engine must
also expose a public ``unload()`` that drops the runtime + avatar cache so
the operator's "释放显存" button actually releases memory.
"""
from __future__ import annotations


# ----- MockEngine -----------------------------------------------------------


def test_mock_engine_describe_lists_a_single_loaded_model():
    # Mock has no weights: the tab still needs a model entry so the operator
    # can see "nothing is loaded, by design" -- and the buttons stay hidden
    # because there's nothing to load or unload.
    from server.engine.mock_engine import MockEngine

    d = MockEngine().describe()
    assert "models" in d, "every engine must surface a models list for the Models tab"
    assert len(d["models"]) == 1
    m = d["models"][0]
    assert m["id"] == "mock_drive"
    assert m["type"] == "drive_runtime"
    assert m["loaded"] is True
    assert m["can_unload"] is False
    assert m["components"] == []


# ----- MuseTalkEngine.describe ----------------------------------------------


def test_musetalk_engine_describe_lists_unloaded_slot_when_no_weights(tmp_path):
    from server.engine.musetalk_engine import MuseTalkEngine

    eng = MuseTalkEngine(models_dir=tmp_path / "missing")
    d = eng.describe()
    assert d["loaded"] is False
    assert d["weights_present"] is False
    assert len(d["models"]) == 1
    m = d["models"][0]
    assert m["id"] == "musetalk_runtime"
    assert m["type"] == "drive_runtime"
    assert m["loaded"] is False
    assert m["can_unload"] is False
    # Component list comes from MuseTalkEngine.WEIGHTS; every component the
    # runtime loads is named here so the operator knows what they're unloading.
    assert "musetalkV15/unet.pth" in m["components"]
    assert "whisper" in m["components"]


def test_musetalk_engine_describe_reports_loaded_with_stub_runtime(tmp_path):
    # ``_ensure_runtime`` is the gate that touches torch; with a stub it
    # returns immediately and ``loaded`` flips to True -- *without* touching a
    # weight, which is the whole point of having a stable describe. We
    # simulate the warmup path by setting ``_runtime`` directly under the
    # lock the engine already holds (real ``warmup()`` would touch torch).
    from server.engine.musetalk_engine import MuseTalkEngine

    class _StubRuntime:
        def __init__(self):
            self.device = "stub"

    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    d_before = eng.describe()
    assert d_before["models"][0]["loaded"] is False

    with eng._lock:
        eng._runtime = _StubRuntime()
        eng._loaded_at = "2026-10-17T00:00:00Z"
    d_after = eng.describe()
    assert d_after["models"][0]["loaded"] is True
    assert d_after["models"][0]["can_unload"] is True
    assert d_after["models"][0]["loaded_at"] == "2026-10-17T00:00:00Z"


def test_musetalk_engine_unload_drops_runtime_and_avatar_cache(tmp_path, monkeypatch):
    # The operator's "释放显存" button calls ``unload()``: it must drop the
    # loaded runtime, clear the per-avatar cache (latents reference the
    # runtime's device), and call into the CUDA-cache helper so the freed
    # memory is reclaimable immediately. A second describe() reports
    # ``loaded=False`` so the UI flips back to "加载".
    from collections import OrderedDict

    from server.engine.musetalk_engine import MuseTalkEngine

    class _StubRuntime:
        pass

    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    eng._runtime = _StubRuntime()
    eng._loaded_at = "2026-10-17T00:00:00Z"
    eng._avatars = OrderedDict()
    eng._avatars["a"] = object()  # signal that something was cached
    monkeypatch.setattr(eng, "_ensure_runtime", lambda: _StubRuntime())

    eng.unload()

    assert eng._runtime is None
    assert eng._loaded_at is None
    assert eng._avatars == OrderedDict()
    assert eng.describe()["models"][0]["loaded"] is False
    assert eng.describe()["models"][0]["loaded_at"] is None


# ----- size estimation -----------------------------------------------------


def test_musetalk_model_size_is_sum_of_weight_files_on_disk(tmp_path):
    # The Models tab's 显存占用 column shows the on-disk footprint as a
    # stable, reproducible lower bound (the runtime's actual VRAM peak is
    # higher; the operator reads the VRAM card for that). Sums every weight
    # file under models_dir that the runtime actually loads.
    from server.engine.musetalk_engine import MuseTalkEngine

    sizes = {
        "musetalkV15/unet.pth": 100,
        "musetalkV15/musetalk.json": 50,
        "sd-vae/config.json": 10,
        "sd-vae/diffusion_pytorch_model.bin": 5,
        "whisper/config.json": 1,
        "whisper/pytorch_model.bin": 4,
        "whisper/preprocessor_config.json": 1,
    }
    models_dir = tmp_path / "m"
    for rel, size in sizes.items():
        path = models_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)

    eng = MuseTalkEngine(models_dir=models_dir)
    d = eng.describe()
    assert d["models"][0]["estimated_bytes"] == sum(sizes.values())


# ----- load() after unload() round-trip -----------------------------------


def test_musetalk_engine_exposes_load_after_unload(tmp_path, monkeypatch):
    # The Models tab's ``load`` button calls engine.load(), and the real
    # engine only had a private warmup() before this -- a re-load after
    # unload() hit a 400 with the message "no weights to load". The endpoint
    # is supposed to look up engine.load(); the engine must expose one, and
    # calling it must recreate the runtime so describe() flips back to
    # ``loaded=True``.
    import asyncio

    from server.engine.musetalk_engine import MuseTalkEngine

    eng = MuseTalkEngine(models_dir=tmp_path / "m")
    # Stub out _ensure_runtime so the test runs without torch weights;
    # the real call site touches torch and would otherwise blow up.
    class _StubRuntime:
        device = "stub"

    states = {"n": 0}

    def fake_ensure():
        # Mirror what the real ``_ensure_runtime`` does: build the
        # _Runtime and stash it on self._runtime under the lock. The test
        # asserts on ``eng._runtime`` directly, so the side effect is the
        # part that matters.
        states["n"] += 1
        with eng._lock:
            eng._runtime = _StubRuntime()
            eng._loaded_at = "2026-10-17T00:00:00Z"
        return _StubRuntime()

    monkeypatch.setattr(eng, "_ensure_runtime", fake_ensure)
    # warmup() -> _submit -> _warmup_sync on the worker. We don't want to
    # spin that worker up here; replace warmup with the literal sequence
    # the dispatcher would otherwise await.
    async def fake_warmup():
        eng._ensure_runtime()

    monkeypatch.setattr(eng, "warmup", fake_warmup)

    eng.unload()
    assert eng._runtime is None

    # The Models tab POSTs /api/admin/models/{id}/load, which awaits
    # engine.load(). Calling it must recreate _runtime.
    asyncio.run(eng.load())
    assert eng._runtime is not None
    assert eng.describe()["models"][0]["loaded"] is True

