"""Tunable thresholds for ``server.avatar.quality.assess``.

The four module constants are also exposed as admin tunables; the operator
widening ``min_face_width_px`` should silence the "small face" warning on a
borderline portrait, etc. We pin the override path here so the knob actually
changes behaviour, not just the audit row.
"""
import numpy as np

from server.avatar.quality import assess


def _landmarks(w=200, cx=360, top=300, chin=620, nose_x=None, nose_y=None):
    pts = np.zeros((478, 2), dtype=np.float32)
    from server.avatar.landmarks import CHIN, LEFT_CHEEK, NOSE_TIP, RIGHT_CHEEK

    pts[LEFT_CHEEK] = (cx - w / 2, (top + chin) / 2)
    pts[RIGHT_CHEEK] = (cx + w / 2, (top + chin) / 2)
    pts[CHIN] = (cx, chin)
    pts[NOSE_TIP] = (nose_x if nose_x is not None else cx, nose_y if nose_y is not None else (top + chin) / 2)
    return pts


def _canvas(texture=40):
    h, w = 1280, 720
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[280:660, 250:470] = (180, 150, 140)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, texture + 1, (380, 220, 1), dtype=np.int16)
    region = rgb[280:660, 250:470].astype(np.int16) + noise
    rgb[280:660, 250:470] = np.clip(region, 0, 255).astype(np.uint8)
    return rgb


def test_default_thresholds_warn_small_face():
    rgb = _canvas()
    out = assess(rgb, _landmarks(w=110))
    assert any("脸部" in w and "宽度" in w for w in out)


def test_widening_min_face_width_silences_small_face():
    rgb = _canvas()
    pts = _landmarks(w=110)
    # Default warns; with the threshold pushed below the detected face width, the
    # warning goes away (the threshold itself is included in the message).
    out = assess(rgb, pts, thresholds={"min_face_width_px": 50.0})
    assert not any("宽度" in w for w in out)


def test_widening_min_nose_ratio_silences_pose_warning():
    # A clearly turned nose with the default threshold: warns.
    pts = _landmarks(w=200, nose_x=360 + 80)
    assert any("正脸" in w for w in assess(_canvas(), pts))

    # With the ratio threshold pushed below the measured value, no warning.
    out = assess(_canvas(), pts, thresholds={"min_nose_ratio": 0.0})
    assert not any("正脸" in w for w in out)


def test_widening_min_sharpness_silences_blur_warning():
    rgb = _canvas(texture=0)  # uniform face region -> low Laplacian variance
    pts = _landmarks()
    assert any("清晰" in w for w in assess(rgb, pts))

    out = assess(rgb, pts, thresholds={"min_sharpness": 0.0})
    assert not any("清晰" in w for w in out)


def test_unknown_threshold_key_ignored():
    # Operators gain knobs over time; an unknown key is a config typo, not a
    # crash -- the unknown is silently dropped and the rest still apply.
    rgb = _canvas()
    out = assess(rgb, _landmarks(w=110), thresholds={"min_face_width_px": 50.0, "future_knob": 999.0})
    assert not any("宽度" in w for w in out)
