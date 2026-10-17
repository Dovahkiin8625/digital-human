"""Tunables for the idle-video H.264 encoder.

``video._encode_h264`` and ``build_idle_video`` previously baked ``crf=18`` and
``preset="veryfast"``. Both are now overrides: real gains live at lower crf and
slower presets at the cost of file size and CPU, and operators should be able
to pick them without a code change.
"""
import json
import types
from pathlib import Path

import numpy as np

from server.avatar import video
from server.avatar.video import build_idle_video

CANVAS = (720, 1280)
SETTINGS = types.SimpleNamespace(
    fps=25, avatar_max_video_frames=375, frame_width=720, frame_height=1280
)
_KNOWN_CRF = 16
_KNOWN_PRESET = "medium"


def _solid(n):
    return [np.full((CANVAS[1], CANVAS[0], 3), i, np.uint8) for i in range(n)]


class _FakeCap:
    def __init__(self, frames, fps=25.0):
        self._frames, self._fps, self._i = frames, fps, 0

    def isOpened(self):
        return True

    def get(self, prop):
        return self._fps if prop == 5 else len(self._frames)

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

    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, path):
        return _FakeCap(self.frames)


class _Recorder:
    instances: list = []

    def __init__(self):
        _Recorder.instances.append(self)
        self.calls: list[dict] = []

    def __call__(self, path, frames, fps, crf=None, preset=None):
        self.calls.append(
            {"crf": crf, "preset": preset, "fps": fps, "n_frames": len(frames)}
        )
        Path(path).write_bytes(b"FAKEMP4")


def test_idle_video_passes_through_crf_and_preset(monkeypatch, tmp_path):
    _Recorder.instances = []
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(_solid(10)))
    recorder = _Recorder()
    monkeypatch.setattr(video, "_encode_h264", recorder)
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": [0, 2, 4, 6]}))

    build_idle_video(
        d,
        d / "source.mp4",
        SETTINGS,
        crf=_KNOWN_CRF,
        preset=_KNOWN_PRESET,
    )

    assert recorder.calls, "encoder was never called"
    call = recorder.calls[0]
    assert call["crf"] == _KNOWN_CRF
    assert call["preset"] == _KNOWN_PRESET


def test_idle_video_defaults_when_no_override(monkeypatch, tmp_path):
    _Recorder.instances = []
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(_solid(4)))
    recorder = _Recorder()
    monkeypatch.setattr(video, "_encode_h264", recorder)
    d = tmp_path / "av2"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": [0, 1]}))

    build_idle_video(d, d / "source.mp4", SETTINGS)

    assert recorder.calls, "encoder was never called"
    call = recorder.calls[0]
    assert call["crf"] == 18
    assert call["preset"] == "veryfast"
