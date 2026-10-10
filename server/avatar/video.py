"""Video decoding for video-driven avatars.

Torch-free and cv2-lazy: `cv2` exists only in the GPU venv, so `_cv2()` is
imported at call time and the default `.venv` suite stubs it. `decode_frames`
is the single decoder shared by the preparation pipeline and the engine's load
path -- both call it with identical arguments so both see identical frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


class VideoError(Exception):
    pass


def is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in VIDEO_SUFFIXES


def _cv2():
    import cv2

    return cv2


def probe_video(path: Path) -> dict:
    """``{"fps", "frame_count", "duration"}``, or raise `VideoError`."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("cannot open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise VideoError("video has no readable frames")
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def _letterbox(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    # Lazily imported: musetalk_pipeline imports this module (lazily) too, and a
    # module-level import here would close the cycle.
    from server.avatar.musetalk_pipeline import fit_to_canvas

    return np.asarray(fit_to_canvas(Image.fromarray(rgb), width, height))


def decode_frames(
    path: Path, target_fps: int, max_frames: int, canvas: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Decode `path` to canvas-space RGB frames at `target_fps`, by timestamp.

    Returns ``(frames[N,H,W,3] uint8, source_fps)``. Picking the source frame
    nearest ``k/target_fps`` keeps motion at natural speed whatever the source
    fps is; where the source is slower than the target, frames repeat.
    """
    cv2 = _cv2()
    info = probe_video(path)
    src_fps = info["fps"]
    n_out = min(max_frames, int(round(info["duration"] * target_fps)))
    if n_out <= 0:
        raise VideoError("video is shorter than a single output frame")

    # source index -> the output frames that want it (a slow source feeds many)
    last = info["frame_count"] - 1
    wanted: dict[int, list[int]] = {}
    for k in range(n_out):
        # Clamped to the last decoded frame: the final output timestamp can round
        # one past the end when the source is slower than the target.
        src = min(int(round(k * src_fps / target_fps)), last)
        wanted.setdefault(src, []).append(k)

    width, height = canvas
    frames = np.zeros((n_out, height, width, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("cannot open video")
        src_i = 0
        filled = 0
        while filled < n_out:
            ok, frame = cap.read()
            if not ok:
                break
            for k in wanted.get(src_i, ()):
                frames[k] = _letterbox(np.ascontiguousarray(frame[:, :, ::-1]), width, height)
                filled += 1
            src_i += 1
    finally:
        cap.release()
    if filled < n_out:
        # Probed N frames but decoded fewer: keep what actually came out.
        frames = frames[:filled]
    return frames, src_fps
