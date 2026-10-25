"""The `lite` model's face geometry.

The numbers here are not free parameters: the visual model was trained
against exactly this geometry, so a change to any of them means re-training.
These tests pin them.

``cv2`` lives only in the runtime venv, so the whole module skips there.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from server.lite import geometry  # noqa: E402


def _landmarks() -> np.ndarray:
    """110 landmarks with the three the bbox uses in the right places.

    Index 1 and 31 are the eye corners (the horizontal span), 52 is above
    them (the top edge).
    """
    points = np.zeros((110, 2), dtype=np.int32)
    points[1] = (100, 200)
    points[31] = (300, 210)
    points[52] = (200, 150)
    return points


def test_the_face_box_is_a_square_from_the_landmarks():
    bbox = geometry.compute_face_bbox(_landmarks())

    x1, y1, x2, y2 = bbox
    assert x1 == 100 and x2 == 300
    assert y1 == 150
    # Forced square: the side is the horizontal span.
    assert (x2 - x1) == (y2 - y1) == 200


def test_a_degenerate_face_box_is_refused():
    points = _landmarks()
    points[31] = (100, 210)  # the two eye corners coincide

    with pytest.raises(ValueError):
        geometry.compute_face_bbox(points)


def test_the_crop_is_the_documented_size():
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    crop = geometry.crop_face(frame, geometry.compute_face_bbox(_landmarks()))

    assert crop.shape == (geometry.FACE_CROP_SIZE, geometry.FACE_CROP_SIZE, 3)


def test_the_inner_square_is_centred_in_the_crop():
    crop = np.zeros((geometry.FACE_CROP_SIZE, geometry.FACE_CROP_SIZE, 3), dtype=np.uint8)

    inner = geometry.extract_inner(crop)

    assert inner.shape == (geometry.FACE_INNER_SIZE, geometry.FACE_INNER_SIZE, 3)
    # The border is the difference between the two sizes, split evenly.
    assert geometry.FACE_CROP_SIZE - geometry.FACE_INNER_SIZE == 2 * geometry.FACE_BORDER


def test_the_model_input_is_the_reference_and_the_masked_face():
    inner = np.full((geometry.FACE_INNER_SIZE, geometry.FACE_INNER_SIZE, 3), 200, dtype=np.uint8)

    model_input = geometry.build_model_input(inner)

    assert model_input.shape == (6, geometry.FACE_INNER_SIZE, geometry.FACE_INNER_SIZE)
    assert model_input.dtype == np.float32
    # Channel 0..2: the reference face, untouched.
    assert np.allclose(model_input[:3], 200 / 255.0)
    # Channels 3..5: the same face with the synthesis region blanked.
    assert model_input[3:].min() == 0.0
    assert model_input[3:].max() > 0.0
    # The input must not modify the frame it was given.
    assert np.all(inner == 200)


def test_the_prediction_is_pasted_over_the_face_box():
    frame = np.full((1280, 720, 3), 10, dtype=np.uint8)
    bbox = (100, 150, 300, 350)
    crop = geometry.crop_face(frame, bbox)
    prediction = np.full((geometry.FACE_INNER_SIZE, geometry.FACE_INNER_SIZE, 3), 250, dtype=np.uint8)

    geometry.paste_prediction(frame, prediction, crop, bbox, (200, 200))

    # The face region carries the prediction; the rest of the frame does not.
    assert frame[200, 200, 0] == 250
    assert frame[10, 10, 0] == 10
