"""The face geometry: the crop, the model's six-channel input, the paste-back.

Needs cv2, so it runs in the runtime venv and skips elsewhere.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dh_server.wav2lip import geometry  # noqa: E402


def _frame(width: int = 400, height: int = 600) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 30  # BGR: a blue field
    frame[100:300, 50:250] = (10, 200, 90)
    return frame


def test_the_crop_is_the_face_box_resized_to_96():
    frame = _frame()
    crop = geometry.crop_face(frame, (50, 100, 250, 300))
    assert crop.shape == (geometry.FACE_SIZE, geometry.FACE_SIZE, 3)
    # The crop is the box, not a square around it.
    assert tuple(crop[0, 0]) == (10, 200, 90)


def test_a_box_outside_the_frame_is_an_error():
    with pytest.raises(ValueError):
        geometry.crop_face(_frame(), (0, 0, 0, 0))


def test_the_model_input_is_the_masked_face_then_the_original():
    crop = np.full((geometry.FACE_SIZE, geometry.FACE_SIZE, 3), 255, dtype=np.uint8)
    inputs = geometry.build_model_input(crop)
    assert inputs.shape == (6, geometry.FACE_SIZE, geometry.FACE_SIZE)
    assert inputs.dtype == np.float32
    assert inputs.max() == pytest.approx(1.0)

    # Channels 0-2: the masked face -- the lower half is blacked out, which
    # is the "synthesise this" signal the network reads.
    assert np.all(inputs[0, geometry.FACE_SIZE // 2 :, :] == 0.0)
    assert np.all(inputs[0, : geometry.FACE_SIZE // 2, :] == 1.0)
    # Channels 3-5: the face itself.
    assert np.all(inputs[3] == 1.0)


def test_the_model_input_does_not_touch_the_crop():
    crop = np.full((geometry.FACE_SIZE, geometry.FACE_SIZE, 3), 200, dtype=np.uint8)
    geometry.build_model_input(crop)
    assert np.all(crop == 200)


def test_the_paste_back_writes_the_prediction_over_the_box():
    frame = _frame()
    prediction = np.full((geometry.FACE_SIZE, geometry.FACE_SIZE, 3), 7, dtype=np.uint8)
    geometry.paste_prediction(frame, prediction, (50, 100, 250, 300))
    # The box is now the prediction, resized to the box.
    assert tuple(frame[150, 150]) == (7, 7, 7)
    # The rest of the frame is untouched.
    assert tuple(frame[10, 10]) == (30, 0, 0)
