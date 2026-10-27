"""Contract tests for the video decoder.

cv2 lives only in the GPU venv, so `_cv2` is stubbed with a fake capture that
plays a list of solid-colour frames. A solid frame survives the letterbox
resize untouched, so the centre pixel identifies which source frame was picked.
"""
import numpy as np
import pytest

from server.avatar import video
from server.avatar.video import VideoError, decode_frames, is_video_suffix, probe_video

CANVAS = (720, 1280)


class _FakeCap:
    def __init__(self, frames, fps, opened=True):
        self._frames = frames
        self._fps = fps
        self._opened = opened
        self._i = 0

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == _FakeCV2.CAP_PROP_FPS:
            return self._fps
        if prop == _FakeCV2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frames, fps=25.0, opened=True):
        self.frames = frames
        self.fps = fps
        self.opened = opened

    def VideoCapture(self, path):
        # A fresh capture per call: probe() and decode_frames() each open their own.
        return _FakeCap(self.frames, self.fps, self.opened)


def _solid(n, fps=25.0):
    # frame i is a solid grey equal to i, at canvas size so letterbox is a no-op
    return [np.full((CANVAS[1], CANVAS[0], 3), i % 256, np.uint8) for i in range(n)], fps


def _install(monkeypatch, frames, fps=25.0, opened=True):
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(frames, fps, opened))


def test_is_video_suffix():
    assert is_video_suffix(".mp4")
    assert is_video_suffix(".MOV")
    assert not is_video_suffix(".png")
    assert not is_video_suffix("")


def test_probe_reports_fps_count_duration(monkeypatch):
    frames, fps = _solid(50, 25.0)
    _install(monkeypatch, frames, fps)
    info = probe_video("v.mp4")
    assert info["fps"] == 25.0
    assert info["frame_count"] == 50
    assert info["duration"] == pytest.approx(2.0)


def test_probe_raises_on_unopenable(monkeypatch):
    _install(monkeypatch, [], opened=False)
    with pytest.raises(VideoError):
        probe_video("v.mp4")


def test_decode_picks_the_nearest_source_frame_by_timestamp(monkeypatch):
    # 50 fps source, 25 fps target -> keep source frames 0,2,4,...
    frames, _ = _solid(50, 50.0)
    _install(monkeypatch, frames, 50.0)
    out, src_fps = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert src_fps == 50.0
    assert out.shape == (25, CANVAS[1], CANVAS[0], 3)
    assert out[0, 640, 360, 0] == 0
    assert out[1, 640, 360, 0] == 2
    assert out[2, 640, 360, 0] == 4


def test_decode_duplicates_frames_when_upsampling(monkeypatch):
    # 10 fps source, 25 fps target: several outputs share one source frame.
    frames, _ = _solid(10, 10.0)
    _install(monkeypatch, frames, 10.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert out.shape[0] == 25
    assert out[0, 640, 360, 0] == 0
    assert out[1, 640, 360, 0] == 0  # 1*10/25 = 0.4 -> rounds to source 0


def test_decode_caps_at_max_frames(monkeypatch):
    frames, _ = _solid(100, 25.0)
    _install(monkeypatch, frames, 25.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=10, canvas=CANVAS)
    assert out.shape[0] == 10


def test_decode_letterboxes_off_size_frames(monkeypatch):
    frames = [np.full((360, 640, 3), 200, np.uint8) for _ in range(25)]
    _install(monkeypatch, frames, 25.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert out.shape[1:] == (CANVAS[1], CANVAS[0], 3)
    assert tuple(out[0, 5, 5]) == (16, 16, 24)  # letterbox bar
