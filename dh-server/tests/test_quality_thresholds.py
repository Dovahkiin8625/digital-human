"""The quality thresholds ``dh_server.avatar.quality.assess`` warns on.

They are module constants: the operator does not tune them per avatar, so
there is nothing to override. The warnings themselves are what matters --
a small face, a turned head and a blurry face each get flagged, and a
good portrait gets flagged for nothing.
"""
import numpy as np

from dh_server.avatar.quality import assess


def _landmarks(w=200, cx=360, top=300, chin=620, nose_x=None, nose_y=None):
    pts = np.zeros((478, 2), dtype=np.float32)
    from dh_server.avatar.landmarks import CHIN, LEFT_CHEEK, NOSE_TIP, RIGHT_CHEEK

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


def test_small_face_warns():
    rgb = _canvas()
    out = assess(rgb, _landmarks(w=110))
    assert any("脸部" in w and "宽度" in w for w in out)


def test_turned_nose_warns():
    # A clearly turned nose with the default threshold: warns.
    pts = _landmarks(w=200, nose_x=360 + 80)
    assert any("正脸" in w for w in assess(_canvas(), pts))


def test_blurry_face_warns():
    rgb = _canvas(texture=0)  # uniform face region -> low Laplacian variance
    assert any("清晰" in w for w in assess(rgb, _landmarks()))


def test_a_good_portrait_is_not_flagged():
    # The point of the thresholds: a normal portrait produces no warnings
    # at all, so the ones that do appear mean something.
    out = assess(_canvas(), _landmarks())
    assert out == []
