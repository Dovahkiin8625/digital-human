"""Video decoding for video-driven avatars.

Torch-free and cv2-lazy: `cv2` exists only in the GPU venv, so `_cv2()` is
imported at call time and the default `.venv` suite stubs it. `decode_frames`
is the single decoder shared by the preparation pipeline and the engine's load
path -- both call it with identical arguments so both see identical frames.
"""
from __future__ import annotations

import threading
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


#: Per-avatar build locks, so two first-requests for the same idle clip
#: serialize: the loser waits and reuses the file instead of reading a
#: half-written one or building a second copy.
_idle_locks: dict[str, threading.Lock] = {}


def build_idle_video(avatar_dir: Path, source: Path, settings) -> Path:
    """Render ``<avatar_dir>/idle.mp4``: the kept source frames ping-ponged.

    Pure CPU and torch-free. The frame set is exactly what the engine's idle
    track plays -- decode with the prepare pipeline's arguments and select the
    frames named by ``musetalk/meta.json``'s ``kept`` -- then write
    ``seq + seq[-2:0:-1]``, the same 2N-2 period ``cycle_index`` walks. The
    file is built via ``idle.mp4.tmp`` and atomically replaced so a reader can
    never see a partial clip.
    """
    import json
    import os

    avatar_dir = Path(avatar_dir)
    meta_path = avatar_dir / "musetalk" / "meta.json"
    out = avatar_dir / "idle.mp4"
    lock = _idle_locks.setdefault(str(avatar_dir), threading.Lock())
    with lock:
        if out.exists():
            return out
        try:
            meta = json.loads(meta_path.read_text())
            kept = [int(i) for i in meta["kept"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VideoError(f"cannot read kept frames from {meta_path}") from exc
        frames, _ = decode_frames(
            source,
            target_fps=settings.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if not kept or max(kept) >= len(frames):
            raise VideoError(
                f"kept frames are outside the decoded sequence "
                f"(kept={len(kept)}, decoded={len(frames)})"
            )
        seq = frames[kept]
        clip = np.concatenate((seq, seq[-2:0:-1]), axis=0)
        tmp = out.with_name("idle.mp4.tmp")
        cv2 = _cv2()
        writer = cv2.VideoWriter(
            str(tmp),
            cv2.VideoWriter_fourcc(*"mp4v"),
            settings.fps,
            (settings.frame_width, settings.frame_height),
        )
        if not writer.isOpened():
            writer.release()
            tmp.unlink(missing_ok=True)
            raise VideoError("cannot open idle video writer")
        try:
            for rgb in clip:
                writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
            writer.release()
            os.replace(tmp, out)
        except VideoError:
            raise
        except Exception as exc:  # noqa: BLE001 -- disk full / dir removed mid-build
            writer.release()
            tmp.unlink(missing_ok=True)
            raise VideoError(f"cannot write idle clip to {out}") from exc
    return out
