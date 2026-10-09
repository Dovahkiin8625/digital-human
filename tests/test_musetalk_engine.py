from pathlib import Path

import numpy as np
import pytest

from server.engine.base import AvatarPack, EngineUnavailable, create_engine
from server.engine.musetalk_engine import MuseTalkEngine


def _pack(tmp_path):
    from PIL import Image

    img = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (5, 5, 5)).save(img)
    return AvatarPack(avatar_id="a", root=tmp_path, source_image_path=img, config={})


def test_prepare_raises_without_weights(tmp_path):
    engine = MuseTalkEngine(models_dir=tmp_path / "missing")
    with pytest.raises(EngineUnavailable):
        import asyncio

        asyncio.run(engine.prepare(_pack(tmp_path)))


def test_describe_and_construct():
    engine = MuseTalkEngine(models_dir=Path("data/models/musetalk"))
    d = engine.describe()
    assert d["name"] == "musetalk"
    assert d["models_dir"] == str(Path("data/models/musetalk"))
    assert isinstance(d["gpu"], bool)


def test_create_engine_musetalk():
    engine = create_engine("musetalk")
    assert isinstance(engine, MuseTalkEngine)
    assert engine.name == "musetalk"


@pytest.mark.asyncio
async def test_drive_pipeline_with_stubbed_model(tmp_path, monkeypatch):
    engine = MuseTalkEngine(models_dir=tmp_path / "m")
    monkeypatch.setattr(engine, "_extract_audio_features", lambda audio, sr: np.zeros((len(audio) // 1920, 16), dtype=np.float32))
    monkeypatch.setattr(engine, "_run_unet", lambda feats, i: np.zeros((256, 256, 3), dtype=np.uint8))
    monkeypatch.setattr(engine, "_load_model_if_needed", lambda: None)
    frames = await engine.drive(_pack(tmp_path), np.zeros(48000, dtype=np.float32), 48000)
    assert len(frames) == 25
    assert frames[0].image.shape == (1280, 720, 3)
    assert frames[0].image.dtype == np.uint8
    assert frames[0].pts == 0.0
    assert abs(frames[1].pts - 0.04) < 1e-6
