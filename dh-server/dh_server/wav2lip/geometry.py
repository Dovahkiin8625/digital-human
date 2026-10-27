"""The face crop and the model's input, for the Wav2Lip model.

Wav2Lip works on a **square 96x96 crop of the face**. The crop is not
square-preserving: the face box is resized to 96x96 whatever its aspect
ratio, exactly as upstream does, so the network sees the faces it was trained
on.

The network's input is **six channels**: the face with its lower half
blacked out, then the face itself. The blacked-out half is the "this is what
you have to synthesise" signal -- the mouth is what the model is asked to
invent, and the rest of the face is the reference it paints around.

The output is the whole 96x96 face, and upstream pastes *all* of it back over
the box it came from. The upper half is a near-copy of the input, so this is
what upstream's videos look like.

``cv2`` is imported lazily: it only exists in the runtime venv, and the
torch-free ``.venv`` must still be able to import this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- typing only
    Bbox = tuple[int, int, int, int]

#: The square the network sees.
FACE_SIZE = 96


def _cv2():
    import cv2

    return cv2


def crop_face(img: np.ndarray, bbox: "Bbox") -> np.ndarray:
    """Cut the face box out of ``img`` and resize it to 96x96.

    ``bbox`` is ``(x1, y1, x2, y2)`` in the frame's pixel coordinates.
    """
    cv2 = _cv2()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError(f"face bbox {bbox} is outside the frame")
    return cv2.resize(region, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)


def build_model_input(crop: np.ndarray) -> np.ndarray:
    """The network's 6-channel input from one face crop.

    Channel order: the masked face (3) then the original face (3), each
    ``float32`` in ``[0, 1]``, shaped ``[6, 96, 96]``.

    ``crop`` is the ``[96, 96, 3]`` uint8 crop and is **not** modified.
    """
    masked = crop.copy()
    masked[FACE_SIZE // 2 :] = 0
    stacked = np.concatenate((masked, crop), axis=2).astype(np.float32) / 255.0
    return np.ascontiguousarray(stacked.transpose(2, 0, 1))


def paste_prediction(
    frame: np.ndarray,
    prediction: np.ndarray,
    bbox: "Bbox",
) -> None:
    """Write the predicted face back into ``frame`` (in place).

    ``prediction`` is the ``[96, 96, 3]`` network output at the model's
    resolution; it is resized to the face box and written over it.
    """
    cv2 = _cv2()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    patch = cv2.resize(prediction, (x2 - x1, y2 - y1))
    frame[y1:y2, x1:x2] = patch
