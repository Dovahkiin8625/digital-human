"""Contract tests for the idle-video generator.

cv2 lives only in the GPU venv, so it is stubbed with a fake capture (solid
source frames); the H.264 encoder seam ``_encode_h264`` is stubbed with a
recorder for the structural tests, while one test exercises the real PyAV
encoder and decodes the artifact back (the browser-playback contract).
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


class _Encoder:
    """Recorder standing in for the real H.264 encoder seam."""

    instances: list["_Encoder"] = []

    def __init__(self, fail: bool = False):
        self.path = None
        self.fps = None
        self.frames: list[np.ndarray] = []
        self.fail = fail
        _Encoder.instances.append(self)

    def __call__(self, path, frames, fps):
        if self.fail:
            raise RuntimeError("encoder exploded")
        self.path = path
        self.fps = fps
        for frame in frames:
            self.frames.append(np.array(frame))
        # The real encoder materialises the file; emulate so the exists() reuse
        # check in build_idle_video works.
        Path(path).write_bytes(b"FAKEMP4")


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, path):
        return _FakeCap(self.frames, 25.0)


def _solid(n):
    return [np.full((CANVAS[1], CANVAS[0], 3), i, np.uint8) for i in range(n)]


def _avatar_dir(tmp_path, kept):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": kept}))
    return d


@pytest.fixture(autouse=True)
def _reset_encoders():
    _Encoder.instances = []
    yield


def _install(monkeypatch, frames, encoder=None):
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(frames))
    # Default to the recorder; the real-encoder test passes the real function.
    monkeypatch.setattr(video, "_encode_h264", encoder or _Encoder())


def test_idle_clip_is_h264_yuv420p_and_keeps_frames(monkeypatch, tmp_path):
    # Browser playback contract: Chromium ships no MPEG-4 Part 2 decoder, so
    # cv2's mp4v clips fetched fine but rendered as a silent black player
    # (MediaError NO_SUPPORTED_STREAMS). Encode for real via the PyAV path and
    # decode the artifact back to prove the ping-pong frames survived.
    import av

    values = [20, 90, 160, 230]
    frames = [np.full((CANVAS[1], CANVAS[0], 3), v, np.uint8) for v in values]
    d = _avatar_dir(tmp_path, [0, 1, 2, 3])
    _install(monkeypatch, frames, video._encode_h264)
    out = build_idle_video(d, d / "source.mp4", SETTINGS)

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        assert codec.name == "h264"
        assert codec.format.name == "yuv420p"
        assert codec.width == CANVAS[0] and codec.height == CANVAS[1]
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    # 2N-2 ping-pong period.
    assert len(decoded) == 6
    centres = [int(f[640, 360, 0]) for f in decoded]
    assert centres == pytest.approx([20, 90, 160, 230, 160, 90], abs=12)


def test_idle_video_writes_kept_frames_ping_pong(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2, 4, 6])
    _install(monkeypatch, _solid(10))
    out = build_idle_video(d, d / "source.mp4", SETTINGS)
    assert out == d / "idle.mp4"
    encoder = _Encoder.instances[0]
    assert encoder.fps == SETTINGS.fps
    # seq (4 frames) + seq[-2:0:-1] (2 frames) = 6 frames, 2N-2 period.
    centre = [int(f[640, 360, 0]) for f in encoder.frames]
    assert centre == [0, 2, 4, 6, 4, 2]
    assert not (d / "idle.tmp.mp4").exists()


def test_tmp_clip_keeps_an_mp4_suffix(monkeypatch, tmp_path):
    # The muxer is picked from the filename extension: a tmp path ending in
    # anything but .mp4 makes the container refuse to open, and a leftover tmp
    # must never be visible to readers.
    d = _avatar_dir(tmp_path, [0, 1])
    _install(monkeypatch, _solid(5))
    build_idle_video(d, d / "source.mp4", SETTINGS)
    tmp_path_used = Path(_Encoder.instances[0].path)
    assert tmp_path_used.suffix == ".mp4"
    assert tmp_path_used.parent == d
    assert not tmp_path_used.exists()


def test_idle_video_passes_rgb_frames_to_the_encoder(monkeypatch, tmp_path):
    # The capture is BGR (as real cv2 delivers): B=30 G=20 R=10, so decoded RGB
    # is (10,20,30). The PyAV encoder takes rgb24 directly, so it must receive
    # the decoded RGB -- no BGR flip at the boundary.
    frames = [np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)]
    frames[0][:, :] = (30, 20, 10)  # BGR: cv2 captures arrive as BGR
    d = _avatar_dir(tmp_path, [0])
    _install(monkeypatch, frames)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert tuple(_Encoder.instances[0].frames[0][640, 360]) == (10, 20, 30)


def test_idle_video_is_built_once_then_reused(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2])
    _install(monkeypatch, _solid(5))
    build_idle_video(d, d / "source.mp4", SETTINGS)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert len(_Encoder.instances) == 1


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


def test_encoder_failure_is_video_error_and_drops_tmp(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 1])

    def boom(path, frames, fps):
        # Emulate an encoder dying mid-clip after materialising a partial tmp
        # file (disk full, dir removed, libx264 init failure, ...).
        Path(path).write_bytes(b"HALF")
        raise RuntimeError("disk full")

    _install(monkeypatch, _solid(5), boom)
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
    # The unlink must remove the partial clip so it is never served.
    assert not (d / "idle.tmp.mp4").exists()
    assert not (d / "idle.mp4").exists()


def test_malformed_kept_is_video_error(monkeypatch, tmp_path):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": 5}))
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
