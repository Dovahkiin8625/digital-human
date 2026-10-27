import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

# Every test in this file bakes overlays, and the bake needs opencv: the
# expression layers warp a patch with cv2 and the blink warps the mesh.
# opencv-python has no cp314 wheel, so the torch-free `.venv` cannot
# install it -- the guard makes the file skip cleanly there instead of
# erroring, exactly as the torch-dependent files do (README: 带 torch 的
# 文件另有 pytest.importorskip("torch") 兜底). The same guard as
# test_overlay_warp.py's scipy one, for the same reason.
pytest.importorskip("cv2")

from dh_server.engine.base import AvatarPack
from dh_server.overlay.landmarks import (
    LEFT_EYE_RING, RIGHT_EYE_RING, Landmarks, boxes_from_points,
)
from dh_server.overlay.prep import preload_overlays
from dh_server.overlay.rig import (
    BROW_L, BROW_R, FACEOVAL, LIP_ALL, LIP_LOWER, LIP_UPPER, expression_offset,
    gaze_offset,
)


class _FakeDetector:
    """Returns hand-crafted landmarks; matches the contract of mediapipe detector."""

    def __init__(self):
        self.calls = 0

    def detect(self, rgb: np.ndarray) -> Landmarks:
        self.calls += 1
        h, w = rgb.shape[:2]
        cx, cy = w // 2, int(h * 0.4)
        pts = _face_mesh(cx, cy)
        # The region boxes come from the mesh, exactly as the real detector
        # derives them (`boxes_from_points`), so the fixture's boxes and the
        # rig's boxes can never drift apart: the mouth box the rig bans is
        # the mouth box the detector reports.
        # The eyelid rings track the eyes _draw_face actually renders, so a
        # layer baked from them lands on the pupil rather than beside it.
        # mediapipe always supplies both; see LEFT_EYE_RING.
        return Landmarks(
            points=pts,
            confidence=0.9,
            eye_l_ring=pts[list(LEFT_EYE_RING)],
            eye_r_ring=pts[list(RIGHT_EYE_RING)],
            **boxes_from_points(pts),
        )


def _ellipse_ring(cx, cy, rx, ry, n=16):
    """A 16-point closed contour, standing in for an eyelid ring."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1).astype(np.float32)


def _arc(cx, cy, rx, ry, a0, a1, n):
    """n points along an ellipse arc, angles in degrees."""
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1).astype(np.float32)


#: The mouth's angles, per lip-ring index, in mediapipe's own order: the
#: upper ring runs from the image-left corner (61) over the top of the
#: upper lip (0) to the image-right corner (409); the lower ring runs from
#: the image-left corner (61) under the bottom of the lower lip (17) to
#: the image-right corner (291).
_UPPER_ANGLES = (180, 200, 220, 240, 255, 270, 285, 300, 320, 360)
_LOWER_ANGLES = (180, 160, 140, 120, 100, 90, 80, 60, 40, 20, 0)

#: The fixture's cheek landmarks: three per side, in the band the rig's
#: cheek rule can move -- inside the cheek box, outboard of the mouth box
#: and above the lip line.
#:
#: Without them the rule finds NOTHING on this face: the face oval at the
#: mouth's height is ~130 px further out, so the cheek box is empty and a
#: smile moves only the brows and the lower lids. Deleting the rig's cheek
#: rule outright would then leave every expression test green.
CHEEK_L = (200, 201, 202)
CHEEK_R = (203, 204, 205)
#: The colour of the marks drawn at the cheek landmarks, so the movement is
#: visible in the picture (the cheek itself is flat skin, and a stretched
#: flat region is invisible).
CHEEK_MARK = (70, 110, 170)


def _cheek_points(cx: int, cy: int) -> np.ndarray:
    """The six cheek landmarks: a short chain down each cheek, outboard of
    the mouth's corners."""
    left = [(cx - 95, cy + 40), (cx - 100, cy + 70), (cx - 105, cy + 100)]
    right = [(cx + 95, cy + 40), (cx + 100, cy + 70), (cx + 105, cy + 100)]
    return np.asarray(left + right, dtype=np.float32)


def _face_mesh(cx: int, cy: int) -> np.ndarray:
    """A 478-point face mesh laid out like the face `_draw_face` draws.

    The eye rings are ON the eyes, and every eye is ringed by a halo of
    static points. The halo is not decoration: without neighbours above
    the eyelid, Delaunay ties the lid straight to the face oval, and a
    blink then drags half the face down with the lid.

    The layout follows mediapipe's own grouping, so the rig's landmark
    groups (brows, lips, face oval) point at the right places instead of
    at random vertices of the outline. The 10 iris points are on the
    pupils, as in the refined model, so the eye's interior is triangulated
    like a real mesh and not as a bare fan.

    Every group is written to ITS OWN indices. The groups used to be
    handed out of the free-index pool in order, which put the brow ring's
    coordinates on the halo's vertices and the halo's on the brow ring's
    -- `rig.BROW_L` then described the RIGHT side of the face, and the
    lip ring wrapped the whole head. Task 4 displaces the brows through
    BROW_L/BROW_R and bans the mouth through LIP_ALL, so that swap made
    every assertion about the wrong pixels (and inverted `sad`'s
    inner/outer brow asymmetry, which is split by distance to the
    midline).
    """
    pts = np.zeros((478, 2), dtype=np.float32)
    taken: set[int] = set()

    def put(ids, coords):
        for i, c in zip(ids, coords):
            pts[i] = c
            taken.add(int(i))

    # The eyelids: exactly on the eyes the face drawing renders.
    put(LEFT_EYE_RING, _ellipse_ring(cx - 60, cy - 55, 30, 25))
    put(RIGHT_EYE_RING, _ellipse_ring(cx + 60, cy - 55, 30, 25))
    # The irises, on the pupils the drawing renders.
    put(range(468, 473), _iris(cx - 60, cy - 55))
    put(range(473, 478), _iris(cx + 60, cy - 55))
    # The face oval is RESERVED before the other groups are allocated:
    # FACEOVAL's indices are scattered through the mesh, and a group
    # allocated over them would be clobbered when the oval is written
    # (and the clobbered point would then be displaced as if it were an
    # oval point). The rig's groups are reserved for the same reason.
    taken |= set(FACEOVAL)
    taken |= set(BROW_L) | set(BROW_R) | set(LIP_UPPER) | set(LIP_LOWER)
    taken |= set(CHEEK_L) | set(CHEEK_R)
    # A halo of static points around each eye.
    halo_l = _arc(cx - 60, cy - 55, 52, 44, 0, 360, 12)
    halo_r = _arc(cx + 60, cy - 55, 52, 44, 0, 360, 12)

    free = [i for i in range(478) if i not in taken]
    for g in (halo_l, halo_r):
        take, free = free[:len(g)], free[len(g):]
        for i, c in zip(take, g):
            pts[i] = c
            taken.add(int(i))
    # The rig's groups, each on its own indices. The lower lip is written
    # first: index 61 is in both lip rings, and both put it on the same
    # mouth corner.
    put(LIP_LOWER, _lip(cx, cy, _LOWER_ANGLES, 40.0))
    put(LIP_UPPER, _lip(cx, cy, _UPPER_ANGLES, 30.0))
    # The brows arch over the eyes they belong to, on the same side of the
    # face as the eye. `BROW_L` is the image-left brow.
    put(BROW_L, _arc(cx - 60, cy - 100, 32, 14, 180, 360, len(BROW_L)))
    put(BROW_R, _arc(cx + 60, cy - 100, 32, 14, 180, 360, len(BROW_R)))
    # The cheeks, in the band the rig's cheek rule can move.
    put(CHEEK_L + CHEEK_R, _cheek_points(cx, cy))
    # The face oval, and then the rest of the outline.
    put(FACEOVAL, _oval(cx, cy, 200, 250))
    rest = [i for i in range(478) if i not in taken]
    ang = np.linspace(0, 2 * np.pi, len(rest), endpoint=False) + 0.01
    # A shade inside the oval, so no filler point lands on top of an oval
    # point: a coincident pair makes a zero-area Delaunay triangle, and the
    # warp's affine for it is singular.
    for i, a in zip(rest, ang):
        pts[i] = (cx + 197 * np.cos(a), cy + 246 * np.sin(a))
    return pts


def _lip(cx: int, cy: int, angles, ry: float) -> np.ndarray:
    """The mouth's ellipse: the drawn mouth's width (60 px) and the given
    vertical radius, centred where `_draw_face` puts the mouth."""
    """Points on the mouth's ellipse: the drawn mouth's width (60 px) and the
    given vertical radius, centred where `_draw_face` puts the mouth."""
    a = np.radians(np.asarray(angles, dtype=float))
    return np.stack(
        [cx + 60 * np.cos(a), (cy + 70) + ry * np.sin(a)], axis=1
    ).astype(np.float32)


def _oval(cx: int, cy: int, rx: float, ry: float) -> np.ndarray:
    """The face oval, walked from the top of the forehead down the image-right
    side to the chin and back up the image-left side.

    The chin is landmark 152, and it has to be the bottom of the oval: the
    mouth exclusion's landmark fallback reaches down to it, and a chin on
    the jaw's side would put the exclusion off the mouth entirely.
    """
    n = len(FACEOVAL)
    half = n // 2 + 1                 # the top -> the chin
    step = 180.0 / (half - 1)
    angles = [270.0 - step * k for k in range(half)]
    angles += [90.0 - step * j for j in range(1, n - half + 1)]
    a = np.radians(np.asarray(angles))
    return np.stack(
        [cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1
    ).astype(np.float32)


def _iris(cx, cy, r=9.0, n=5):
    """n points on the iris: the centre plus four around it."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    return np.vstack([ring, [cx, cy]]).astype(np.float32)


def _small_eye_mesh(cx, cy, rx=6.0, ry=3.0):
    """A 478-point mesh whose eye openings are 2*rx by 2*ry px.

    The small-eye case: the reference avatar's opening is 27.8 x 11.4 px,
    and a face further away gives a smaller one still. The inward feather
    costs 2r px of the opening's height, so its radius has to follow the
    opening down.
    """
    pts = np.zeros((478, 2), dtype=np.float32)
    taken: set[int] = set()

    def put(ids, coords):
        for i, c in zip(ids, coords):
            pts[i] = c
            taken.add(int(i))

    put(LEFT_EYE_RING, _ellipse_ring(cx, cy, rx, ry))
    put(RIGHT_EYE_RING, _ellipse_ring(cx + 40, cy, rx, ry))
    put(range(468, 473), _iris(cx, cy, r=max(1.2, ry * 0.6)))
    put(range(473, 478), _iris(cx + 40, cy, r=max(1.2, ry * 0.6)))
    taken |= set(FACEOVAL)
    halo_l = _arc(cx, cy, rx * 3, ry * 3, 0, 360, 12)
    halo_r = _arc(cx + 40, cy, rx * 3, ry * 3, 0, 360, 12)
    free = [i for i in range(478) if i not in taken]
    for g in (halo_l, halo_r):
        take, free = free[:len(g)], free[len(g):]
        for i, c in zip(take, g):
            pts[i] = c
            taken.add(int(i))
    put(FACEOVAL, _arc(cx + 20, cy, 60, 40, 0, 360, len(FACEOVAL)))
    rest = [i for i in range(478) if i not in taken]
    ang = np.linspace(0, 2 * np.pi, len(rest), endpoint=False) + 0.01
    for i, a in zip(rest, ang):
        pts[i] = (cx + 20 + 60 * np.cos(a), cy + 40 * np.sin(a))
    return pts


def _draw_small_face(path: Path, size=(240, 240), eye=(80, 100),
                     rx=6.0, ry=3.0):
    """A face with two small eyes, for the small-eye blink test."""
    img = Image.new("RGB", size, (220, 180, 140))
    d = ImageDraw.Draw(img)
    for ex in (eye[0], eye[0] + 40):
        d.ellipse((ex - rx, eye[1] - ry, ex + rx, eye[1] + ry), fill=(255, 255, 255))
        d.ellipse((ex - 2, eye[1] - 2, ex + 2, eye[1] + 2), fill=(0, 0, 0))
    img.save(path)


def _draw_face(path: Path, size=(720, 1280)):
    """The face the fixture's landmarks describe.

    Eyes at (cx-60, cy-55) and (cx+60, cy-55), a mouth arc under them at
    (cx, cy+70), and a brow arc over each eye. The brows are drawn because
    the expression layer moves them: without a brow in the picture there is
    nothing to see move.
    """
    img = Image.new("RGB", size, (10, 10, 10))
    d = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, int(size[1] * 0.4)
    d.ellipse((cx - 200, cy - 250, cx + 200, cy + 250), fill=(220, 180, 140))
    for ex in (cx - 60, cx + 60):
        d.arc((ex - 32, cy - 114, ex + 32, cy - 86), 180, 360,
              fill=(90, 60, 45), width=6)
    d.ellipse((cx - 90, cy - 80, cx - 30, cy - 30), fill=(255, 255, 255))
    d.ellipse((cx + 30, cy - 80, cx + 90, cy - 30), fill=(255, 255, 255))
    d.ellipse((cx - 70, cy - 65, cx - 50, cy - 45), fill=(0, 0, 0))
    d.ellipse((cx + 50, cy - 65, cx + 70, cy - 45), fill=(0, 0, 0))
    d.arc((cx - 60, cy + 30, cx + 60, cy + 110), 0, 180, fill=(120, 30, 30), width=4)
    # The cheek marks, on the cheek landmarks: the cheek is flat skin, and
    # a stretched flat region is invisible, so the movement needs a mark.
    for x, y in _cheek_points(cx, cy):
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=CHEEK_MARK)
    img.save(path)


def _drawn_brightness(img: np.ndarray, pts) -> float:
    """The mean brightness of the drawn pixels under `pts`."""
    vals = [float(img[int(round(y)), int(round(x))].mean()) for x, y in pts]
    return float(np.mean(vals))


def test_fixture_geometry_is_coherent(tmp_path):
    """The rig's landmark groups must sit on the features the drawing
    renders.

    Task 4 displaces the brows through BROW_L/BROW_R and derives the mouth
    exclusion from LIP_ALL. With the groups on the wrong features -- the
    brows swapped left for right, the lips wrapped around the eyes -- every
    assertion about the expression layer is about the wrong pixels, and
    `sad`'s inner/outer brow asymmetry (which is split by distance to the
    face midline) is inverted. This pins the geometry so it cannot
    silently regress again.
    """
    src = tmp_path / "src.png"
    _draw_face(src)
    img = np.array(Image.open(src).convert("RGB"))
    lm = _FakeDetector().detect(img)
    p = np.asarray(lm.points)

    eye_l = p[list(LEFT_EYE_RING)]
    eye_r = p[list(RIGHT_EYE_RING)]
    lips = p[list(LIP_ALL)]
    brow_l = p[list(BROW_L)]
    brow_r = p[list(BROW_R)]
    midline_x = float(eye_l[:, 0].mean() + eye_r[:, 0].mean()) / 2.0

    # The eyes are where the drawing renders them.
    assert eye_l[:, 0].min() == pytest.approx(270.0, abs=1)
    assert eye_l[:, 0].max() == pytest.approx(330.0, abs=1)
    assert eye_l[:, 1].min() == pytest.approx(432.0, abs=1)
    assert eye_l[:, 1].max() == pytest.approx(482.0, abs=1)
    assert eye_r[:, 0].min() == pytest.approx(390.0, abs=1)
    assert eye_r[:, 0].max() == pytest.approx(450.0, abs=1)

    # Each brow sits above its own eye, on its own side of the face.
    assert brow_l[:, 1].max() < eye_l[:, 1].min()
    assert brow_r[:, 1].max() < eye_r[:, 1].min()
    assert brow_l[:, 0].max() < midline_x
    assert brow_r[:, 0].min() > midline_x

    # The lips are below the eyes and inside the drawn mouth.
    assert lips[:, 1].min() > max(eye_l[:, 1].max(), eye_r[:, 1].max())
    assert lips[:, 0].min() >= 300 and lips[:, 0].max() <= 420
    assert lips[:, 1].min() >= 542 and lips[:, 1].max() <= 622

    # ...and the brows are actually DRAWN under the brow rings (the skin
    # around them averages 180).
    assert _drawn_brightness(img, brow_l) < 150
    assert _drawn_brightness(img, brow_r) < 150

    # The detector's mouth box -- the one the rig bans and the one the
    # exclusion is derived from -- does not reach the eyes.
    mx, my, mw, mh = lm.mouth_box
    assert my > eye_l[:, 1].max()
    assert (mx, my, mw, mh) == boxes_from_points(p)["mouth_box"]

    # The chin is below the lips: the exclusion's landmark fallback reaches
    # down to it.
    assert p[152, 1] > lips[:, 1].max()

    # The eyelid has static neighbours above it: without them Delaunay ties
    # the lid straight to the face oval and the blink drags the face down.
    above = [i for i in range(len(p))
             if i not in set(LEFT_EYE_RING)
             and p[i, 1] < eye_l[:, 1].min()
             and eye_l[:, 0].min() <= p[i, 0] <= eye_l[:, 0].max()]
    assert above, "the eyelid needs static neighbours above it"


def test_preload_overlays_photo_writes_pack_block(tmp_path):
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    block = preload_overlays(pack, src, detector=_FakeDetector())
    assert block["version"] == 1
    assert set(block["tokens"]["expression"]) == {"neutral", "smile", "sad"}
    assert set(block["tokens"]["gaze"]) == {"center", "left", "right"}
    # 9 expression × gaze PNGs + 1 blink = 10 assets on disk
    overlays = tmp_path / "overlays"
    pngs = list(overlays.glob("*.png"))
    assert len(pngs) == 10
    # landmarks.json present
    assert (overlays / "landmarks.json").exists()


def test_preload_overlays_returns_block_referenced_from_pack(tmp_path):
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    block = preload_overlays(pack, src, detector=_FakeDetector())
    # Every relative path in the block must exist on disk.
    root = Path(pack.root)
    for expr, gazes in block["assets"]["expression"].items():
        for gaze, rel in gazes.items():
            assert (root / rel).exists(), (expr, gaze, rel)
    assert (root / block["assets"]["blink"]).exists()
    assert (root / block["landmarks"]).exists()


def test_preload_overlays_landmarks_match_face(tmp_path):
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    block = preload_overlays(pack, src, detector=_FakeDetector())
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    assert "points" in raw
    assert raw["confidence"] == pytest.approx(0.9)


def test_preload_overlays_idempotent_overwrites(tmp_path):
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_FakeDetector())
    preload_overlays(pack, src, detector=_FakeDetector())
    # No duplicate or stray files; second run is a clean rewrite.
    overlays = tmp_path / "overlays"
    assert len(list(overlays.glob("*.png"))) == 10


# ---------------------------------------------------------------------------
# Overlay bake quality: the three defects that made a baked blink look like a
# pasted-on rectangle rather than a closed eye.
# ---------------------------------------------------------------------------


def test_gauss_feather_softens_the_mask_border():
    """The mask is already cropped to its box, so blurring it in place mirrors
    the 255 interior outward (cv2's default BORDER_REFLECT_101) and the alpha
    steps 0 -> 255 in a single pixel. Every overlay then shows a hard seam."""
    from dh_server.overlay.prep import _gauss_feather

    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[:] = 255
    out = _gauss_feather(mask, radius=8)

    assert out[0, 0] < 200, "border must be softened, not saturated"
    ramp = out[:, 20]
    assert ((ramp > 10) & (ramp < 245)).sum() >= 3, "expected a soft alpha ramp"


def _bake_blink(tmp_path):
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_FakeDetector())
    rgb = np.array(Image.open(src).convert("RGB")).astype(np.float32)
    layer = np.array(
        Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA")
    ).astype(np.float32)
    return rgb, layer, _FakeDetector().detect(rgb).eye_l_box


def test_blink_layer_is_not_a_pure_translation_of_the_source(tmp_path):
    """A closed eye must change the pixels. The old bake slid the eye box up by
    (bh - bh//3)//2 and re-cropped the same window, which made the layer
    byte-identical to the source at that offset -- the eye never closed."""
    rgb, layer, (x, y, bw, bh) = _bake_blink(tmp_path)
    box = layer[y:y + bh, x:x + bw, :3]

    best = min(
        float(((box - rgb[y + dy:y + dy + bh, x:x + bw]) ** 2).mean())
        for dy in range(-bh, bh + 1)
        if 0 <= y + dy and y + dy + bh <= rgb.shape[0]
    )
    assert best > 0.0, "blink layer is a pure shift of the source; the eye never closes"


def test_blink_layer_shrinks_the_pupil(tmp_path):
    """The behavioural claim behind the shift fix: compositing the blink layer
    must actually remove the open pupil."""
    rgb, layer, (x, y, bw, bh) = _bake_blink(tmp_path)
    a = layer[y:y + bh, x:x + bw, 3:4] / 255.0
    out = (layer[y:y + bh, x:x + bw, :3] * a
           + rgb[y:y + bh, x:x + bw] * (1 - a))

    def dark_px(img):
        return int((img.max(axis=2) < 60).sum())

    assert dark_px(out) < dark_px(rgb[y:y + bh, x:x + bw]) * 0.5, (
        "pupil should be mostly covered once the eye is closed"
    )


def _lower_lid_y(ring) -> float:
    """The lower lid's y at the ring's centre x -- where a blink puts the lid."""
    ring = np.asarray(ring, dtype=np.float32)
    mid_y = float(ring[:, 1].mean())
    lower = ring[ring[:, 1] >= mid_y]
    lower = lower[np.argsort(lower[:, 0])]
    centre_x = float(ring[:, 0].mean())
    return float(np.interp(centre_x, lower[:, 0], lower[:, 1]))


def test_blink_layer_collapses_the_eyelid_mesh_onto_the_lower_lid(tmp_path):
    """Blink now descends the upper-lid vertices rather than squeezing the
    whole box: the brow and the cheek under the eye stay where they are."""
    import json as _json
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_FakeDetector())
    raw = _json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    ring = raw["eye_l_ring"]
    assert ring, "the detector must supply eyelid rings for the blink to follow"
    layer = np.array(Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA"))
    base = np.array(Image.open(src).convert("RGB")).astype(np.float32)

    # (a) nothing above the eye opening is touched. The alpha has to stop
    # at the ring's top edge: the brow above it is not part of a blink,
    # and a feather that reaches past the opening smears the skin the
    # warp stretched there.
    top = int(min(p[1] for p in ring))
    assert layer[top - 4:top, :, 3].max() == 0, (
        "the lid above the eye must not be touched"
    )

    # (b) the eye region ITSELF is rewritten. Without this the assertion
    # above passes on the previous implementation too -- it already masked
    # to the ring -- so it would pin nothing about the mesh change.
    a = layer[:, :, 3:4] / 255.0
    composed = layer[:, :, :3] * a + base * (1 - a)
    cx = int(np.mean([q[0] for q in ring]))
    cy = int(np.mean([q[1] for q in ring]))
    eye = (slice(cy - 12, cy + 12), slice(cx - 18, cx + 18))
    assert np.abs(composed[eye] - base[eye]).max() > 25, (
        "the eye opening must visibly change; a layer identical to the source "
        "here is the bug this whole change exists to fix"
    )

    # (c) and the eye is SHUT: nothing of the open pupil survives in the
    # middle of the opening. A box squeeze leaves the pupil squashed into
    # the centre of the opening; a mesh blink either covers it with the
    # falling lid or squeezes it onto the lower lid's line.
    x0 = int(min(q[0] for q in ring))
    x1 = int(max(q[0] for q in ring))
    y0 = int(min(q[1] for q in ring))
    seam_y = _lower_lid_y(ring)
    band = (slice(y0 + 2, max(y0 + 3, int(seam_y) - 4)), slice(x0, x1 + 1))
    dark_src = int((base[band].max(axis=2) < 60).sum())
    dark_out = int((composed[band].max(axis=2) < 60).sum())
    assert dark_src >= 20, "the fixture must have a pupil to close"
    assert dark_out <= max(4, dark_src // 10), (
        "the open pupil must not survive in the middle of the opening"
    )


def test_blink_alpha_touches_nothing_outside_the_eye_opening(tmp_path):
    """The feather has to grow inward.

    `cover` -- the pixels the moved triangles reached -- is not the alpha:
    it runs 24 px above the opening on this fixture, and `warped` still
    holds the ORIGINAL content wherever a triangle vacated, so an
    outward feather composites a rim of pre-blink pixels around the eye.
    """
    import cv2
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_FakeDetector())
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    layer = np.array(Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA"))

    opening = np.zeros(layer.shape[:2], dtype=np.uint8)
    for key in ("eye_l_ring", "eye_r_ring"):
        pts = np.round(np.asarray(raw[key], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(opening, [pts], 255)
    outside = (layer[:, :, 3] > 0) & (opening == 0)
    assert not outside.any(), (
        "the blink layer must not touch anything outside the eye opening"
    )


def test_blink_closes_the_eye_on_the_base_468_point_mesh(tmp_path):
    """The refined model carries the iris inside the eye opening; the base
    468-point model does not. Both have to end up with the eye shut."""
    import dataclasses
    src = tmp_path / "src.png"
    _draw_face(src)

    class _BaseDetector(_FakeDetector):
        def detect(self, rgb):
            lm = super().detect(rgb)
            pts = lm.points[:468]
            return dataclasses.replace(
                lm, points=pts,
                eye_l_ring=pts[list(LEFT_EYE_RING)],
                eye_r_ring=pts[list(RIGHT_EYE_RING)],
            )

    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_BaseDetector())
    rgb = np.array(Image.open(src).convert("RGB")).astype(np.float32)
    layer = np.array(
        Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA")
    ).astype(np.float32)
    x, y, bw, bh = 262, 424, 76, 66
    a = layer[y:y + bh, x:x + bw, 3:4] / 255.0
    out = (layer[y:y + bh, x:x + bw, :3] * a
           + rgb[y:y + bh, x:x + bw] * (1 - a))

    def dark_px(img):
        return int((img.max(axis=2) < 60).sum())

    assert dark_px(out) < dark_px(rgb[y:y + bh, x:x + bw]) * 0.5, (
        "the eye must be shut on the base mesh too"
    )


def test_blink_still_covers_a_small_eye_opening(tmp_path):
    """A 12x6 px opening is the small-eye case, and it used to vanish
    silently: the inward feather costs 2r px of the opening's height, so a
    fixed 2 px radius ate a third of it and the alpha never saturated."""
    import dataclasses
    src = tmp_path / "src.png"
    _draw_small_face(src)

    class _SmallDetector(_FakeDetector):
        def detect(self, rgb):
            lm = super().detect(rgb)
            pts = _small_eye_mesh(80, 100)
            return dataclasses.replace(
                lm, points=pts,
                eye_l_ring=pts[list(LEFT_EYE_RING)],
                eye_r_ring=pts[list(RIGHT_EYE_RING)],
            )

    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_SmallDetector())
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    layer = np.array(Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA"))
    ring = np.asarray(raw["eye_l_ring"], dtype=np.float32)
    x0, x1 = int(ring[:, 0].min()), int(ring[:, 0].max())
    y0, y1 = int(ring[:, 1].min()), int(ring[:, 1].max())
    opening = layer[y0:y1 + 1, x0:x1 + 1, 3]

    assert opening.size > 0, "the small eye must be in the layer at all"
    assert int(opening.max()) >= 200, (
        f"the blink's alpha must saturate on a small eye: peak was "
        f"{int(opening.max())} over a {x1 - x0 + 1}x{y1 - y0 + 1} px opening"
    )
    assert int((opening >= 200).sum()) >= 0.15 * opening.size, (
        f"the blink must cover a real part of a small eye: "
        f"{int((opening >= 200).sum())}/{opening.size} px reach alpha 200"
    )


def test_blink_ignores_a_partial_eyelid_contour(tmp_path):
    """A contour that lies on the mesh but is shorter than the ring would
    drive a partial blink, masked to a partial polygon. The mesh's own
    ring is used instead."""
    import dataclasses
    src = tmp_path / "src.png"
    _draw_face(src)

    class _PartialDetector(_FakeDetector):
        def detect(self, rgb):
            lm = super().detect(rgb)
            half = lm.eye_l_ring[:8]
            return dataclasses.replace(lm, eye_l_ring=half)

    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_PartialDetector())
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    layer = np.array(Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA"))
    ring = np.asarray(raw["eye_l_ring"], dtype=np.float32)
    x0, x1 = int(ring[:, 0].min()), int(ring[:, 0].max())
    y0, y1 = int(ring[:, 1].min()), int(ring[:, 1].max())
    alpha = layer[:, :, 3]
    assert alpha[y1 - 1, x0:x1 + 1].max() > 0, (
        "a partial contour must not shorten the blink: the lower lid's line "
        "is outside the half-contour the detector supplied"
    )
    assert alpha[y0, x0:x1 + 1].max() > 0, (
        "a partial contour must not shorten the blink at the top either"
    )


def test_blink_ignores_an_eyelid_contour_that_is_not_on_the_mesh(tmp_path):
    """A detector may hand over a contour of its own. Chasing a contour
    that is not on the mesh would move whatever mesh vertex happens to
    be nearest -- possibly on the other side of the face -- and the blink
    would drag the whole face with the eyelid."""
    import dataclasses
    src = tmp_path / "src.png"
    _draw_face(src)
    h, w = 1280, 720
    cx, cy = w // 2, int(h * 0.4)

    class _StrayDetector(_FakeDetector):
        def detect(self, rgb):
            lm = super().detect(rgb)
            stray = _ellipse_ring(cx, cy + 150, 30, 20)
            return dataclasses.replace(lm, eye_l_ring=stray, eye_r_ring=None)

    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_StrayDetector())
    layer = np.array(Image.open(tmp_path / "overlays" / "blink.png").convert("RGBA"))
    alpha = layer[:, :, 3]

    assert alpha[cy + 130:cy + 170, cx - 30:cx + 30].max() == 0, (
        "a contour that is not on the mesh must not steer the blink"
    )
    assert alpha[432:482, 270:330].max() > 200, (
        "the eye must still blink: the mesh's own eyelid ring is used"
    )


# ---------------------------------------------------------------------------
# Task 4: the expression drives the mesh, and it never covers the mouth.
# ---------------------------------------------------------------------------


def _bake(tmp_path):
    """Bake the overlay assets of the fixture face into `tmp_path`."""
    src = tmp_path / "src.png"
    _draw_face(src)
    pack = AvatarPack(avatar_id="a1", root=tmp_path, source_image_path=src, config={})
    preload_overlays(pack, src, detector=_FakeDetector())
    return src


def _layer(tmp_path, name):
    return np.array(Image.open(tmp_path / "overlays" / name).convert("RGBA"))


def _composite(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """The compositor's own arithmetic, so the assertions are about what a
    frame actually looks like after the overlay is applied."""
    a = layer[:, :, 3:4].astype(np.float32) / 255.0
    return (layer[:, :, :3] * a + base * (1 - a)).astype(np.uint8)


def _lip_bbox(tmp_path):
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    lips = np.asarray(raw["points"], dtype=np.float32)[list(LIP_ALL)]
    return (int(lips[:, 0].min()), int(lips[:, 1].min()),
            int(lips[:, 0].max()), int(lips[:, 1].max()))


def test_expression_layer_is_transparent_over_the_mouth(tmp_path):
    """The expression layer is composited over MuseTalk's output, and
    MuseTalk drives the mouth. Measured on a real avatar, the old layer was
    69.8% opaque over the mouth -- so any smile replaced the audio-driven
    mouth with the static face this layer was baked from."""
    _bake(tmp_path)
    x0, y0, x1, y1 = _lip_bbox(tmp_path)
    alpha = _layer(tmp_path, "expr_smile_center.png")[y0:y1 + 1, x0:x1 + 1, 3]
    assert alpha.max() == 0, (
        f"the mouth must be left to MuseTalk, got alpha {alpha.max()}"
    )


def test_expression_layer_holes_out_the_musetalk_mask(tmp_path):
    """With a MuseTalk sidecar the exclusion is the engine's own mask --
    the ground truth for what the engine rewrites.

    The fake mask covers the image-LEFT brow only: the right brow has to
    stay opaque, so the assertion fails if the whole layer is zeroed and
    it fails if the sidecar is ignored and the landmark fallback is used
    instead (the fallback never reaches the brows).
    """
    musetalk = tmp_path / "musetalk"
    musetalk.mkdir()
    Image.fromarray(np.full((40, 60), 255, dtype=np.uint8), mode="L").save(
        musetalk / "mask.png")
    (musetalk / "bbox.json").write_text(json.dumps({
        "bbox": [0, 0, 0, 0],
        "mask_crop_box": [260, 380, 320, 420],
    }))
    _bake(tmp_path)
    alpha = _layer(tmp_path, "expr_smile_center.png")[:, :, 3]
    assert alpha[380:420, 260:320].max() == 0, (
        "the MuseTalk mask must be punched out of the expression layer"
    )
    assert alpha[398:412, 388:452].max() > 0, (
        "the rest of the layer must survive: the mask is a hole, not a "
        "blanket"
    )


def test_the_musetalk_mask_threshold_is_zero(tmp_path):
    """The engine blends the driven face through `mask / 255`, so it
    rewrites every pixel the mask touches, however faintly.

    Thresholding the mask at 127 leaves the ring 0 < mask <= 127 out of the
    exclusion: on the reference avatar that ring is 4 520 px, and the
    expression layer was opaque over ~40 px of it.
    """
    musetalk = tmp_path / "musetalk"
    musetalk.mkdir()
    faint = np.zeros((40, 60), dtype=np.uint8)
    faint[10:30, 20:50] = 3          # barely anything at all
    Image.fromarray(faint, mode="L").save(musetalk / "mask.png")
    (musetalk / "bbox.json").write_text(json.dumps({
        "bbox": [0, 0, 0, 0],
        "mask_crop_box": [260, 380, 320, 420],
    }))
    _bake(tmp_path)
    alpha = _layer(tmp_path, "expr_smile_center.png")[:, :, 3]
    # The mask's nonzero block, in frame coords.
    assert alpha[390:410, 280:310].max() == 0, (
        "a faint mask pixel is still rewritten by the engine"
    )


def test_the_mouth_exclusion_reads_the_production_musetalk_layout(tmp_path):
    """The pipeline writes `musetalk/masks/{i:08d}.png` + `musetalk/coords.json`
    since `2a98589`; the `mask.png` + `bbox.json` layout is dead for every
    avatar built since, and reading only it left production protected by
    the landmark fallback -- a fraction of the mouth.

    The fake mask sits over the LEFT brow, far from the lips, so the
    assertion cannot be satisfied by the landmark fallback.
    """
    from dh_server.overlay.prep import _mouth_exclusion

    musetalk = tmp_path / "musetalk"
    (musetalk / "masks").mkdir(parents=True)
    Image.fromarray(np.full((40, 60), 255, dtype=np.uint8), mode="L").save(
        musetalk / "masks" / "00000000.png")
    (musetalk / "coords.json").write_text(json.dumps([
        {"bbox": [0, 0, 0, 0], "mask_crop_box": [260, 380, 320, 420]},
    ]))
    src = tmp_path / "src.png"
    _draw_face(src)
    img = np.array(Image.open(src).convert("RGB"))
    lm = _FakeDetector().detect(img)
    ex = _mouth_exclusion(tmp_path, lm, img.shape[:2])
    assert ex[380:420, 260:320].all(), (
        "the production sidecar must be read: masks/00000000.png + coords.json"
    )


def test_the_landmark_fallback_exclusion_is_only_the_lips(tmp_path):
    """Without a MuseTalk sidecar there is no live mouth to protect.

    The fallback used to widen the lips' bbox to 2.2x its width and run it
    down to the chin, which swallowed both cheek boxes whole: the cheek
    rule was computed and then thrown away by the exclusion. The fallback
    is now the outer lips' own bbox and nothing else.
    """
    from dh_server.overlay.prep import _mouth_exclusion

    src = tmp_path / "src.png"
    _draw_face(src)
    img = np.array(Image.open(src).convert("RGB"))
    lm = _FakeDetector().detect(img)
    ex = _mouth_exclusion(tmp_path, lm, img.shape[:2])
    lips = np.asarray(lm.points)[list(LIP_ALL)]
    x0, y0 = int(lips[:, 0].min()), int(lips[:, 1].min())
    x1, y1 = int(lips[:, 0].max()), int(lips[:, 1].max())

    assert ex[y0:y1 + 1, x0:x1 + 1].all(), "the lips must be excluded"
    assert not ex[y1 + 1:].any(), "nothing below the lips may be excluded"
    assert not ex[:, :x0].any(), "nothing left of the lips may be excluded"
    assert not ex[:, x1 + 1:].any(), "nothing right of the lips may be excluded"


def test_the_expression_layer_is_not_blanked_over_the_cheeks(tmp_path):
    """The exclusion must not swallow the cheek boxes.

    Measured on the fixture with the old fallback: the exclusion spanned
    x 224-495, y 548-765, which contains both cheek boxes entirely, and
    `alpha > 0` inside the exclusion was 0 px -- so a mock-path avatar's
    expression layer could only ever move brows and lids.

    The cheek box is checked OUTSIDE the lips: the lips themselves are
    excluded on purpose.
    """
    _bake(tmp_path)
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    points = np.asarray(raw["points"], dtype=np.float32)
    from dh_server.overlay.landmarks import boxes_from_points
    boxes = boxes_from_points(points)
    lips = points[list(LIP_ALL)]
    lx0, ly0 = int(lips[:, 0].min()), int(lips[:, 1].min())
    lx1, ly1 = int(lips[:, 0].max()), int(lips[:, 1].max())
    alpha = _layer(tmp_path, "expr_smile_center.png")[:, :, 3]

    mouth = np.zeros(alpha.shape, dtype=bool)
    mouth[ly0:ly1 + 1, lx0:lx1 + 1] = True
    for key in ("cheek_l_box", "cheek_r_box"):
        x, y, bw, bh = boxes[key]
        # The cheek band from the lips' top edge down -- the band the old
        # fallback swallowed whole.
        cheek = np.zeros(alpha.shape, dtype=bool)
        cheek[max(y, ly0):y + bh, x:x + bw] = True
        outside = cheek & ~mouth
        assert outside.any()
        assert alpha[outside].max() > 0, (
            f"the {key} must not be blanked by the mouth exclusion"
        )


def test_gaze_layer_moves_the_iris_not_the_eyelid_ring(tmp_path):
    """Gaze moves the iris; the eyelid ring stays where it is.

    The property is asserted, not a pixel threshold. The old implementation
    translated the whole eye box, so its pupil moved AND the eye opening
    moved with it -- and a "the pupil moved by more than N px" threshold
    cannot tell the two apart, and cannot be carried over to a face of a
    different size (the fixture's iris is 18 px across, the reference
    portrait's 11.3 px).

    So: the pupil travels the distance the rig says it does -- a fraction
    of the iris's own diameter, and therefore a distance that scales with
    the face -- and the eye opening does not move sideways at all.
    """
    src = _bake(tmp_path)
    base = np.array(Image.open(src).convert("RGB"))
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    points = np.asarray(raw["points"], dtype=np.float32)
    travel = gaze_offset(points, "left")               # negative, iris-scaled
    layer = _layer(tmp_path, "expr_neutral_left.png")
    comp = _composite(base, layer)
    eye = (slice(432, 483), slice(270, 331))           # the drawn left eye

    def pupil_x(img):
        return float(np.nonzero(img[eye].max(axis=2) < 60)[1].mean())

    def opening_x(img):
        bright = img[eye].min(axis=2) > 200
        cols = np.flatnonzero(bright.any(axis=0))
        return int(cols.min()) + eye[1].start, int(cols.max()) + eye[1].start

    moved = pupil_x(base) - pupil_x(comp)
    assert moved > 0, "gaze left must carry the pupil toward image-left"
    assert moved == pytest.approx(-travel, rel=0.3), (
        f"the pupil must travel the rig's {abs(travel):.2f} px, it moved "
        f"{moved:.2f}"
    )
    # The eyelid ring did not move, so the eye's own opening did not move.
    # (The old implementation slid the whole eye box 10 px to the left:
    # its opening's right edge went from 330 to 320.)
    assert abs(opening_x(comp)[0] - opening_x(base)[0]) <= 1, (
        "the eye opening must not travel with the iris"
    )
    assert abs(opening_x(comp)[1] - opening_x(base)[1]) <= 1, (
        "the eye opening must not travel with the iris"
    )
    assert layer[:, :, 3].max() == 255


def test_gaze_layer_is_not_a_box_translation(tmp_path):
    """Spec 9.1: the gaze layer is a mesh warp, not the eye box slid.

    The old implementation translated the eye box by a constant 10 px, so
    its layer WAS the source shifted by exactly (-10, 0): the MSE against
    that translation was 0.0. This layer's best match is ~1 200.
    """
    src = _bake(tmp_path)
    base = np.array(Image.open(src).convert("RGB"))
    rgb = _layer(tmp_path, "expr_neutral_left.png")[:, :, :3]
    h, w = base.shape[:2]
    y0, x0, y1, x1 = 432, 270, 483, 331                # the drawn left eye
    best = None
    for dy in range(-10, 11):
        for dx in range(-16, 17):
            if dx == 0 and dy == 0:
                continue
            ys, ye = max(y0, dy), min(y1, h + dy)
            xs, xe = max(x0, dx), min(x1, w + dx)
            if ye <= ys or xe <= xs:
                continue
            mse = float((
                (rgb[ys:ye, xs:xe].astype(np.float32)
                 - base[ys - dy:ye - dy, xs - dx:xe - dx].astype(np.float32)) ** 2
            ).mean())
            best = mse if best is None else min(best, mse)
    assert best is not None and best > 0.0, (
        "the gaze layer is a pure translation of the source; the iris has to "
        "move inside the eye, not the eye box"
    )


def test_the_smile_layer_moves_the_cheeks(tmp_path):
    """Spec 9.1's `test_smile_moves_cheeks_and_brows`, at the layer level.

    The rig's cheek rule can only move a landmark that is inside the cheek
    box, outboard of the mouth box and above the lip line. On this face
    there was NOTHING there -- the face oval is ~130 px further out -- so
    the rule found no target at all, a smile moved only brows and lids,
    and deleting the rule outright would have left every other expression
    test green.

    The fixture now carries three cheek landmarks per side with a mark
    drawn on each (the cheek is flat skin, and a stretched flat region is
    invisible), so the movement reaches the pixels and can be measured.
    """
    src = _bake(tmp_path)
    base = np.array(Image.open(src).convert("RGB"))
    raw = json.loads((tmp_path / "overlays" / "landmarks.json").read_text())
    points = np.asarray(raw["points"], dtype=np.float32)
    boxes = {
        "eye_l_box": tuple(raw["eye_l_box"]),
        "eye_r_box": tuple(raw["eye_r_box"]),
        "mouth_box": tuple(raw["mouth_box"]),
        "cheek_l_box": tuple(raw["cheek_l_box"]),
        "cheek_r_box": tuple(raw["cheek_r_box"]),
    }
    dst, moved = expression_offset(points, expr="smile", boxes=boxes)
    assert set(CHEEK_L) | set(CHEEK_R) <= set(moved), (
        "the rig's cheek rule must find the fixture's cheek landmarks"
    )
    layer = _layer(tmp_path, "expr_smile_center.png")
    comp = _composite(base, layer)

    def mark_centre(img, x0, x1):
        region = img[500:680, x0:x1]
        bluish = region[:, :, 2].astype(int) > region[:, :, 0].astype(int) + 30
        ys, xs = np.nonzero(bluish)
        assert len(xs), "the cheek mark must be in the picture"
        return float(xs.mean() + x0), float(ys.mean() + 500)

    for ids, x0, x1 in ((CHEEK_L, 200, 320), (CHEEK_R, 400, 520)):
        ids = list(ids)
        mesh_dy = float((dst[ids, 1] - points[ids, 1]).mean())
        mesh_dx = float((dst[ids, 0] - points[ids, 0]).mean())
        bx, by = mark_centre(base, x0, x1)
        cx, cy = mark_centre(comp, x0, x1)
        assert by - cy == pytest.approx(-mesh_dy, rel=0.35), (
            f"the cheek must rise by the rig's {-mesh_dy:.1f} px, it rose "
            f"{by - cy:.1f}"
        )
        assert cx - bx == pytest.approx(mesh_dx, rel=0.35), (
            f"the cheek must spread by the rig's {mesh_dx:.1f} px, it spread "
            f"{cx - bx:.1f}"
        )


def test_neutral_layer_is_identical_to_the_source(tmp_path):
    """A zero-displacement bake must not stamp the frame with itself."""
    src = _bake(tmp_path)
    base = np.array(Image.open(src).convert("RGB"))
    layer = _layer(tmp_path, "expr_neutral_center.png")
    assert layer[:, :, 3].max() == 0, (
        "a neutral, centred expression has nothing to composite"
    )
    comp = _composite(base, layer)
    assert np.abs(comp.astype(np.int16) - base.astype(np.int16)).max() <= 1


def test_smile_layer_lifts_the_brow(tmp_path):
    """The expression drives the mesh: the drawn brow goes up, and the
    lip ring's own pixels do not move with it."""
    src = _bake(tmp_path)
    base = np.array(Image.open(src).convert("RGB"))
    layer = _layer(tmp_path, "expr_smile_center.png")
    comp = _composite(base, layer)

    band = (slice(388, 424), slice(258, 342))          # the drawn left brow

    def brow_y(img):
        ys, _ = np.nonzero(img[band].mean(axis=2) < 140)
        return float(ys.mean()) + band[0].start

    assert brow_y(comp) < brow_y(base) - 3.0, (
        f"the smile must lift the brow: {brow_y(base):.1f} -> {brow_y(comp):.1f}"
    )
    # ...and the mouth is left alone: the exclusion means the composite over
    # the drawn mouth is the base, byte for byte.
    mouth = (slice(566, 622), slice(300, 420))
    assert np.abs(comp[mouth].astype(np.int16)
                   - base[mouth].astype(np.int16)).max() <= 1, (
        "the drawn mouth must not be touched by the expression layer"
    )

