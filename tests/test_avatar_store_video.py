"""`save_source` routes by suffix: images decode via PIL, videos via a probe
that also enforces the duration/frame cap at upload time."""
import io

import numpy as np
import pytest
from PIL import Image

from server.avatar.store import AvatarStore
from server.avatar.video import VideoError
from server.core.errors import AppError


def _store(tmp_path, monkeypatch, **extra_env):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    for key, value in extra_env.items():
        monkeypatch.setenv(key, str(value))
    from server.config import get_settings

    get_settings.cache_clear()
    return AvatarStore(tmp_path / "data")


def test_image_still_validated_by_pil(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    ok = tmp_path / "ok.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(ok)
    path = store.save_source("a1", io.BytesIO(ok.read_bytes()), ".png")
    assert path == store.avatar_dir("a1") / "source.png"

    with pytest.raises(AppError) as ei:
        store.save_source("a2", io.BytesIO(b"not an image"), ".png")
    assert ei.value.code == "invalid_image"
    assert ei.value.status == 422
    # A rejected upload gets no Avatar row; the empty avatar dir goes too.
    assert not store.avatar_dir("a2").exists()


def test_video_accepts_a_probeable_file(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": 25.0, "frame_count": 50, "duration": 2.0},
    )
    path = store.save_source("v1", io.BytesIO(b"\x00" * 64), ".mp4")
    assert path == store.avatar_dir("v1") / "source.mp4"
    assert path.exists()


def test_video_rejects_undecodable(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)

    def boom(p):
        raise VideoError("cannot open video")

    monkeypatch.setattr("server.avatar.video.probe_video", boom)
    with pytest.raises(AppError) as ei:
        store.save_source("v2", io.BytesIO(b"\x00" * 64), ".mp4")
    assert ei.value.code == "invalid_video"
    assert ei.value.status == 422
    # The rejected upload must not be left behind: no Avatar row exists to own it.
    assert not store.avatar_dir("v2").exists()


def test_video_rejects_too_long(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": 25.0, "frame_count": 500, "duration": 20.0},
    )
    with pytest.raises(AppError) as ei:
        store.save_source("v3", io.BytesIO(b"\x00" * 64), ".mp4")
    assert ei.value.code == "video_too_long"
    assert ei.value.status == 422
    # The rejected upload must not be left behind: no Avatar row exists to own it.
    assert not (store.avatar_dir("v3") / "source.mp4").exists()
    assert not store.avatar_dir("v3").exists()


def test_stream_over_cap_mid_copy_is_413_and_dir_removed(tmp_path, monkeypatch):
    cap = 1024
    store = _store(tmp_path, monkeypatch, DH_AVATAR_MAX_UPLOAD_BYTES=cap)
    data = b"\x00" * (cap + 1)
    with pytest.raises(AppError) as ei:
        store.save_source("big", io.BytesIO(data), ".png")
    assert ei.value.code == "file_too_large"
    assert ei.value.status == 413
    d = store.avatar_dir("big")
    assert not list(d.glob("source*"))
    # Mid-copy abort: neither the partial file nor the empty dir survives.
    assert not d.exists()
