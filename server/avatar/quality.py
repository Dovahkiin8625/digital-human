"""Advisory source-quality checks for avatar uploads.

MuseTalk resynthesises a face at a fixed 256x256: a face that occupies too few
of the 720x1280 canvas pixels, a strongly turned head, a soft/compressed face,
or a non-9:16 source letterboxed with bars all cap the result *before* any
model runs. These checks surface those causes to the operator as Chinese,
actionable warnings.

They never fail a job -- the pipeline still produces the avatar. Only the
truly unusable cases (no detectable face, undecodable file) are hard errors.

Everything here is numpy/PIL only so the checks run in the base venv and are
unit-testable without a GPU. Landmarks are the mediapipe FaceMesh canvas-space
points produced by `MuseTalkPreparer.landmarks`.
"""
from __future__ import annotations

import numpy as np

from server.avatar.landmarks import CHIN, LEFT_CHEEK, NOSE_TIP, RIGHT_CHEEK

#: Cheek-to-cheek face width on the 720-wide canvas below which the 256x256
#: generation is upscaling more detail than the source held.
MIN_FACE_WIDTH_PX = 160
#: Nearer/farther nose-to-cheek distance ratio below which the head is turned
#: hard enough that the far half of the mouth is poorly constrained.
MIN_NOSE_RATIO = 0.55
#: Variance of the 4-neighbour Laplacian inside the face box below which the
#: face reads as blurry or heavily compressed.
MIN_SHARPNESS = 60.0
#: fit_to_canvas letterbox fill; bars in exactly this colour are ours.
CANVAS_BG = np.array([16, 16, 24], dtype=np.int16)
#: Per-channel tolerance for a pixel counting as letterbox background.
BG_TOL = 10
#: A border line at least this fraction background counts as a bar.
BG_LINE = 0.98
#: The source must occupy at least this fraction of each canvas axis or it is
#: flagged as a non-portrait composition.
MIN_OCCUPANCY = 0.92


def face_width(pts: np.ndarray) -> float:
    return float(pts[RIGHT_CHEEK, 0] - pts[LEFT_CHEEK, 0])


def nose_ratio(pts: np.ndarray) -> float:
    """1.0 for a frontal nose, shrinking as the head turns either way."""
    nose_x = float(pts[NOSE_TIP, 0])
    d_left = abs(nose_x - float(pts[LEFT_CHEEK, 0]))
    d_right = abs(float(pts[RIGHT_CHEEK, 0]) - nose_x)
    if d_left <= 0 or d_right <= 0:
        return 1.0
    return min(d_left, d_right) / max(d_left, d_right)


def face_box(pts: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
    """The same region the pipeline crops: cheeks across, nose-to-chin doubled."""
    nose_y = float(pts[NOSE_TIP, 1])
    chin_y = float(pts[CHIN, 1])
    upper = max(0.0, nose_y - (chin_y - nose_y))
    x1 = max(0, int(round(float(pts[LEFT_CHEEK, 0]))))
    x2 = min(w, int(round(float(pts[RIGHT_CHEEK, 0]))))
    y1 = max(0, int(round(upper)))
    y2 = min(h, int(round(chin_y)))
    return x1, y1, x2, y2


def sharpness(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Variance of the 4-neighbour Laplacian over the face box (focus proxy)."""
    region = rgb[max(0, y1 - 1) : y2 + 1, max(0, x1 - 1) : x2 + 1]
    if region.shape[0] < 4 or region.shape[1] < 4:
        return MIN_SHARPNESS
    gray = region.astype(np.float32).mean(axis=2)
    lap = (
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(lap.var())


def _bar_inset(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Consecutive near-background lines inset from each canvas edge."""
    h, w = mask.shape
    left = right = top = bottom = 0
    while left < w and mask[:, left].mean() >= BG_LINE:
        left += 1
    while right < w - left and mask[:, w - 1 - right].mean() >= BG_LINE:
        right += 1
    while top < h and mask[top, :].mean() >= BG_LINE:
        top += 1
    while bottom < h - top and mask[h - 1 - bottom, :].mean() >= BG_LINE:
        bottom += 1
    return left, right, top, bottom


def assess(rgb: np.ndarray, pts: np.ndarray, thresholds: dict | None = None) -> list[str]:
    """Return human-readable quality warnings for one prepared canvas frame.

    Empty list means nothing worth flagging. Order is fixed (size, pose,
    sharpness, composition) so warnings read identically across runs.

    ``thresholds`` is a dict of admin tunables; missing keys fall back to the
    module constants, so existing call sites keep working untouched.
    """
    min_w = (thresholds or {}).get("min_face_width_px", MIN_FACE_WIDTH_PX)
    min_r = (thresholds or {}).get("min_nose_ratio", MIN_NOSE_RATIO)
    min_s = (thresholds or {}).get("min_sharpness", MIN_SHARPNESS)
    min_o = (thresholds or {}).get("min_occupancy", MIN_OCCUPANCY)

    h, w = rgb.shape[:2]
    warnings: list[str] = []

    width = face_width(pts)
    if width < min_w:
        warnings.append(
            f"人脸在画面中仅约 {width:.0f}px 宽，建议更换脸部宽度 ≥{min_w:.0f}px "
            "的近景竖版素材以提升清晰度"
        )

    ratio = nose_ratio(pts)
    if ratio < min_r:
        warnings.append("人脸偏转角度较大，建议使用双眼平视镜头的正脸素材")

    x1, y1, x2, y2 = face_box(pts, h, w)
    if x2 > x1 and y2 > y1 and sharpness(rgb, x1, y1, x2, y2) < min_s:
        warnings.append("面部区域清晰度偏低，建议使用对焦清晰、无明显压缩噪点的原图")

    bg_mask = (np.abs(rgb.astype(np.int16) - CANVAS_BG).max(axis=2) <= BG_TOL)
    left, right, top, bottom = _bar_inset(bg_mask)
    occ_w = (w - left - right) / w
    occ_h = (h - top - bottom) / h
    if occ_w < min_o or occ_h < min_o:
        warnings.append("素材非 9:16 竖版构图，画面已补黑边，建议裁剪为 9:16 竖版后重新上传")

    return warnings
