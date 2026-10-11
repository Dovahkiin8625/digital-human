"""Source-quality warnings.

The gate is deliberately advisory: warnings never fail the job. These tests pin
the four checks (face size, head turn, facial sharpness, non-9:16 letterbox) on
synthetic canvases and mediapipe-shaped landmark arrays -- no torch, no GPU.
"""
import numpy as np

from server.avatar.quality import assess


def _landmarks(w=200, cx=360, top=300, chin=620, nose_x=None, nose_y=None):
    """A minimal 478-point landmark array with only the indices assess() reads.

    Cheeks at cx +/- w/2, nose on the midline (a frontal face), chin/top set
    explicitly. Unused indices are zeros.
    """
    pts = np.zeros((478, 2), dtype=np.float32)
    from server.avatar.landmarks import CHIN, LEFT_CHEEK, NOSE_TIP, RIGHT_CHEEK

    pts[LEFT_CHEEK] = (cx - w / 2, (top + chin) / 2)
    pts[RIGHT_CHEEK] = (cx + w / 2, (top + chin) / 2)
    pts[CHIN] = (cx, chin)
    pts[NOSE_TIP] = (nose_x if nose_x is not None else cx, nose_y if nose_y is not None else (top + chin) / 2)
    return pts


def _canvas_with_face(face=(180, 150, 140), size=(1280, 720), texture=40):
    """RGB canvas with a textured face-colour rectangle in the middle.

    `texture` is the per-pixel value jitter: 0 is a flat wall (blurry), large
    values read as sharp detail.
    """
    h, w = size
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[280:660, 250:470] = np.asarray(face, dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, texture + 1, (380, 220, 1), dtype=np.int16)
    region = rgb[280:660, 250:470].astype(np.int16) + noise
    rgb[280:660, 250:470] = np.clip(region, 0, 255).astype(np.uint8)
    return rgb


def test_good_frontal_photo_warns_nothing():
    rgb = _canvas_with_face()
    assert assess(rgb, _landmarks()) == []


def test_small_face_warns_about_face_width():
    rgb = _canvas_with_face()
    warnings = assess(rgb, _landmarks(w=110))
    assert any("脸部" in w and "宽度" in w for w in warnings), warnings


def test_turned_head_warns_about_frontal_pose():
    # Nose a quarter of the cheek span from one cheek: a strong yaw.
    pts = _landmarks(w=200, nose_x=360 + 55)
    warnings = assess(_canvas_with_face(), pts)
    assert any("正脸" in w for w in warnings), warnings


def test_flat_face_region_warns_about_sharpness():
    rgb = _canvas_with_face(texture=0)
    warnings = assess(rgb, _landmarks())
    assert any("清晰" in w for w in warnings), warnings


def test_letterboxed_wide_source_warns_about_aspect():
    # 16:9 content centered in the 9:16 canvas leaves fill-colour bars at the
    # sides: occupied width is 720*9/16 = 405 px.
    rgb = np.zeros((1280, 720, 3), dtype=np.uint8)
    rgb[:, :] = (16, 16, 24)
    rgb[160:1120, 157:563] = _canvas_with_face()[160:1120, 157:563]
    warnings = assess(rgb, _landmarks())
    assert any("竖版" in w or "9:16" in w for w in warnings), warnings


def test_warnings_are_unique_and_ordered():
    # Tiny, turned, blurry: one of each, no duplicates.
    rgb = _canvas_with_face(texture=0)
    pts = _landmarks(w=100, nose_x=360 + 40)
    warnings = assess(rgb, pts)
    assert len(warnings) == len(set(warnings))
    assert len(warnings) >= 2
