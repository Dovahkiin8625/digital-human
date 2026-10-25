"""Decoding and letterboxing for the `lite` model.

Frames are handled in **BGR** here, because that is the order FeatherTalk was
trained in: its preprocessing wrote JPEGs with ``cv2.imwrite`` and its
inference read them back with ``cv2.imread``, so the network's first channel
is blue. The engine converts to RGB at the very end, when it hands a frame to
the WebRTC track.

The decoder is a *stream*: a training video is minutes long and holding every
frame of it would be tens of gigabytes, so frames are yielded one at a time
and the caller decides what to keep.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

#: The letterbox colour, in BGR. The MuseTalk pipeline uses RGB (16, 16, 24);
#: same colour, opposite order.
CANVAS_BG = (24, 16, 16)


def _cv2():
    import cv2

    return cv2


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a frame into a width x height canvas, preserving aspect ratio.

    Mirrors ``musetalk_pipeline.fit_to_canvas`` (PIL, RGB) but in OpenCV's
    BGR order and without the PIL round trip -- this runs once per frame of a
    training video.
    """
    cv2 = _cv2()
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = CANVAS_BG
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def stream_frames(
    path: Path,
    target_fps: int,
    canvas: tuple[int, int],
    *,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` on the target-fps grid.

    Output slot ``k`` shows the source frame nearest ``k / target_fps``, so
    the motion keeps its natural speed whatever the source frame rate is --
    the same rule ``server.avatar.video.decode_frames`` uses. The index is
    the *output* index, which is what the pack's coordinates are keyed by.
    """
    cv2 = _cv2()
    width, height = canvas
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            raise ValueError(f"video {path} has no readable frame rate")
        ratio = source_fps / target_fps
        source_index = 0
        out_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            while round(out_index * ratio) == source_index:
                yield out_index, letterbox(frame, width, height)
                out_index += 1
                if max_frames is not None and out_index >= max_frames:
                    return
            source_index += 1
    finally:
        capture.release()


def write_jpeg(path: Path, frame: np.ndarray, quality: int = 95) -> None:
    """Write one frame as a JPEG, in BGR."""
    cv2 = _cv2()
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"could not write {path}")
