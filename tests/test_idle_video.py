"""Contract tests for the idle-video generator.

cv2 lives only in the GPU venv, so it is stubbed with a fake capture (solid
source frames) plus a fake VideoWriter that records the encoded frames; a
centre pixel identifies each source frame.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest

from server.avatar import video
from server.avatar.video import VideoError, build_idle_video

CANVAS = (720, 1280)
SETTINGS = types.SimpleNamespace(
    fps=25, avatar_max_video_frames=375, frame_width=720, frame_height=1280
)


class _FakeCap:
    def __init__(self, frames, fps):
        self._frames, self._fps, self._i = frames, fps, 0

    def isOpened(self):
        return True

    def get(self, prop):
        return self._fps if prop == 5 else len(self._frames)  # CAP_PROP_FPS=5

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


class _Writer:
    instances: list["_Writer"] = []

    def __init__(self, path, fourcc, fps, size, opened=True):
        self.path, self.fps, self.size = path, fps, size
        self.frames: list[np.ndarray] = []
        self.opened = opened
        _Writer.instances.append(self)

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(np.array(frame))

    def release(self):
        # The real writer materialises the file; emulate so the exists() reuse
        # check in build_idle_video works.
        Path(self.path).write_bytes(b"FAKEMP4")


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, path):
        return _FakeCap(self.frames, 25.0)

    def VideoWriter(self, path, fourcc, fps, size):
        return _Writer(path, fourcc, fps, size)

    def VideoWriter_fourcc(self, *a):
        return "mp4v"


def _solid(n):
    return [np.full((CANVAS[1], CANVAS[0], 3), i, np.uint8) for i in range(n)]


def _avatar_dir(tmp_path, kept):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": kept}))
    return d


@pytest.fixture(autouse=True)
def _reset_writers():
    _Writer.instances = []
    yield


def _install(monkeypatch, frames):
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(frames))


def test_idle_video_writes_kept_frames_ping_pong(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2, 4, 6])
    _install(monkeypatch, _solid(10))
    out = build_idle_video(d, d / "source.mp4", SETTINGS)
    assert out == d / "idle.mp4"
    writer = _Writer.instances[0]
    assert writer.size == CANVAS
    # seq (4 frames) + seq[-2:0:-1] (2 frames) = 6 frames, 2N-2 period.
    centre = [int(f[640, 360, 0]) for f in writer.frames]
    assert centre == [0, 2, 4, 6, 4, 2]
    assert not (d / "idle.mp4.tmp").exists()


def test_idle_video_flips_rgb_to_bgr_for_the_writer(monkeypatch, tmp_path):
    # The capture is BGR (as real cv2 delivers): B=30 G=20 R=10, so decoded RGB
    # is (10,20,30). After the decode/encode round trip the writer must receive
    # the same BGR (30,20,10) back.
    frames = [np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)]
    frames[0][:, :] = (30, 20, 10)  # BGR: cv2 captures arrive as BGR
    d = _avatar_dir(tmp_path, [0])
    _install(monkeypatch, frames)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert tuple(_Writer.instances[0].frames[0][640, 360]) == (30, 20, 10)


def test_idle_video_is_built_once_then_reused(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2])
    _install(monkeypatch, _solid(5))
    build_idle_video(d, d / "source.mp4", SETTINGS)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert len(_Writer.instances) == 1


def test_missing_meta_is_video_error(monkeypatch, tmp_path):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)


def test_kept_outside_decoded_range_is_video_error(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [99])
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)


def test_unopenable_writer_is_video_error(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 1])
    _install(monkeypatch, _solid(5))

    class _BrokenCV2(_FakeCV2):
        def VideoWriter(self, path, fourcc, fps, size):
            return _Writer(path, fourcc, fps, size, opened=False)

    monkeypatch.setattr(video, "_cv2", lambda: _BrokenCV2(_solid(5)))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
    assert not (d / "idle.mp4.tmp").exists()


def test_writer_failure_is_video_error_and_drops_tmp(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 1])

    class _BoomWriter(_Writer):
        def write(self, frame):
            raise RuntimeError("disk full")

    class _BoomCV2(_FakeCV2):
        def VideoWriter(self, path, fourcc, fps, size):
            return _BoomWriter(path, fourcc, fps, size)

    monkeypatch.setattr(video, "_cv2", lambda: _BoomCV2(_solid(5)))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
    # The cleanup release() emulates materialising the file at the tmp path;
    # the unlink must remove it again so no half clip is left behind.
    assert not (d / "idle.mp4.tmp").exists()


def test_malformed_kept_is_video_error(monkeypatch, tmp_path):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": 5}))
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
