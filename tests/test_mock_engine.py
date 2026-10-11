import numpy as np
import pytest

from server.engine.base import AvatarPack, create_engine
from server.engine.mock_engine import MockEngine


def _pack(tmp_path):
    from PIL import Image

    img_path = tmp_path / "src.png"
    Image.new("RGB", (720, 1280), (30, 60, 90)).save(img_path)
    return AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=img_path, config={})


@pytest.mark.asyncio
async def test_drive_returns_frames_matching_audio_duration(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    audio = np.zeros(48000, dtype=np.float32)  # 1s
    frames = await engine.drive(pack, audio, 48000)
    assert len(frames) == 25
    assert frames[0].image.shape == (1280, 720, 3)
    assert frames[0].image.dtype == np.uint8
    assert frames[0].pts == 0.0
    assert abs(frames[1].pts - 0.04) < 1e-6


@pytest.mark.asyncio
async def test_drive_is_deterministic(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    audio = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    a = await engine.drive(pack, audio, 48000)
    b = await engine.drive(pack, audio, 48000)
    assert all((x.image == y.image).all() for x, y in zip(a, b))


@pytest.mark.asyncio
async def test_loud_audio_opens_mouth_more(tmp_path):
    engine = MockEngine()
    pack = _pack(tmp_path)
    silent = await engine.drive(pack, np.zeros(4800, dtype=np.float32), 48000)
    loud = await engine.drive(pack, np.ones(4800, dtype=np.float32) * 0.9, 48000)
    assert np.abs(loud[0].image.astype(int) - silent[0].image.astype(int)).sum() > 0


def test_create_engine_factory():
    assert create_engine("mock").__class__.__name__ == "MockEngine"
