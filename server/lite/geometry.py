"""Face geometry for the `lite` model: bbox, crop, mask, paste-back.

A port of the geometry in FeatherTalk's `face_utils.py`, kept here so the
runtime path does not import the vendored module (which pulls in torch) and so
the numbers are stated once, in one place, with the upstream quirks spelled
out.

The numbers are **not** free parameters: the visual model was trained against
exactly this geometry, so changing any of them means re-training.

    FACE_CROP_SIZE   152  the square face crop the landmarks select
    FACE_INNER_SIZE  144  the inner square the network actually sees
    FACE_BORDER        4  the border between the two
    MASK_RECT  (5,5,150,145)  the region blanked out in the "masked" channel

The masked channel is *mostly* black on purpose: it is the network's "this
area is to be synthesised" signal, and the network repaints the whole face
patch, not just the mouth. The mouth is where the loss is concentrated during
training (`dataset_mouth_roi.py`), not where the mask is.

``cv2`` is imported lazily: it only exists in the runtime venv, and the
torch-free `.venv` must still be able to import this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- typing only
    Bbox = tuple[int, int, int, int]

#: The square crop the landmarks select, before the inner square is taken.
FACE_CROP_SIZE = 152
#: The square the network sees.
FACE_INNER_SIZE = 144
#: The border between the two.
FACE_BORDER = 4
#: The coordinate space the mask rectangle is expressed in.
GEOMETRY_BASE_SIZE = 160
#: The rectangle blanked out in the "masked" input channel.
MASK_RECT = (5, 5, 150, 145)

#: Landmark indices the face bbox is derived from: the two eye corners give
#: the horizontal span, the point above them gives the top edge.
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 31
TOP_INDEX = 52


def _cv2():
    import cv2

    return cv2


def compute_face_bbox(landmarks: np.ndarray) -> "Bbox":
    """The square face box around the mouth, from 110 landmarks.

    Horizontal span: the two eye corners. Top edge: the landmark above them.
    The box is forced to be a square whose side is the horizontal span, so the
    crop keeps the face's aspect ratio whatever the head pose.

    ``landmarks`` is ``[110, 2]`` in the *frame's* pixel coordinates.
    """
    xmin = int(landmarks[LEFT_EYE_INDEX][0])
    ymin = int(landmarks[TOP_INDEX][1])
    xmax = int(landmarks[RIGHT_EYE_INDEX][0])
    width = xmax - xmin
    if width <= 0:
        raise ValueError(
            f"face landmarks give a non-positive width ({width}); the face "
            "detector probably failed on this frame"
        )
    ymax = ymin + width
    return xmin, ymin, xmax, ymax


def crop_face(img: np.ndarray, bbox: "Bbox", crop_size: int = FACE_CROP_SIZE) -> np.ndarray:
    """Cut the bbox out of ``img`` and resize it to a square."""
    cv2 = _cv2()
    xmin, ymin, xmax, ymax = bbox
    region = img[ymin:ymax, xmin:xmax]
    if region.size == 0:
        raise ValueError(f"face bbox {bbox} is outside the frame")
    return cv2.resize(region, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def extract_inner(face_crop: np.ndarray, inner_size: int = FACE_INNER_SIZE) -> np.ndarray:
    """The centred inner square of a crop."""
    border = (face_crop.shape[0] - inner_size) // 2
    return face_crop[border : border + inner_size, border : border + inner_size].copy()


def mask_mouth(img: np.ndarray, inner_size: int = FACE_INNER_SIZE) -> np.ndarray:
    """Blank out the region the network has to synthesise (in place).

    Upstream writes this as ``cv2.rectangle(img, rect, ...)`` where ``rect`` is
    built from ``(x, y, width, height)`` -- but ``cv2.rectangle`` reads the
    four numbers as ``(x1, y1, x2, y2)``. The names are wrong, the behaviour
    is what the model was trained with, so it is reproduced verbatim here.
    Changing it would silently change what the network sees.
    """
    cv2 = _cv2()
    scale = inner_size / float(GEOMETRY_BASE_SIZE)
    x, y, width, height = MASK_RECT
    rect = (
        round(x * scale),
        round(y * scale),
        round(width * scale),
        round(height * scale),
    )
    return cv2.rectangle(img, rect, (0, 0, 0), -1)


def build_model_input(inner: np.ndarray) -> np.ndarray:
    """The network's 6-channel input from one face's inner square.

    Channel order: the reference face (3) and the masked face (3), each
    ``float32`` in ``[0, 1]``, shaped ``[6, H, W]``.

    ``inner`` is the ``[H, W, 3]`` uint8 inner square and is **not** modified.
    """
    reference = inner.astype(np.float32) / 255.0
    masked = mask_mouth(inner.copy().astype(np.uint8)).astype(np.float32) / 255.0
    return np.concatenate(
        [reference.transpose(2, 0, 1), masked.transpose(2, 0, 1)], axis=0
    )


def paste_prediction(
    frame: np.ndarray,
    prediction: np.ndarray,
    face_crop: np.ndarray,
    bbox: "Bbox",
    original_size: tuple[int, int],
) -> None:
    """Write the predicted face back into ``frame`` (in place).

    ``prediction`` is the ``[H, W, 3]`` uint8 network output at inner size; it
    replaces the inner square of ``face_crop``, which is then resized back to
    the bbox and pasted over the frame's face region.
    """
    cv2 = _cv2()
    border = FACE_BORDER
    inner_slice = face_crop[
        border : border + FACE_INNER_SIZE,
        border : border + FACE_INNER_SIZE,
    ]
    inner_slice[:] = prediction

    crop_width, crop_height = original_size
    face_crop = cv2.resize(face_crop, (crop_width, crop_height))
    xmin, ymin, xmax, ymax = bbox
    frame[ymin:ymax, xmin:xmax] = face_crop
