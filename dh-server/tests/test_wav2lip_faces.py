"""The face box geometry: upstream's padding and smoothing.

Both are pure functions of the boxes, so they are asserted without the
detector and without torch.
"""
from __future__ import annotations

from dh_server.wav2lip import faces as wav2lip_faces


def test_the_padding_adds_room_for_the_chin():
    """Upstream pads the bottom by 10 pixels: the mouth is at the bottom of
    the face, and a crop that cuts it off leaves the model nothing to paint."""
    padded = wav2lip_faces.pad_box((100, 100, 300, 400), shape=(600, 400))
    assert padded == (100, 100, 300, 410)


def test_the_padding_is_clipped_to_the_frame():
    padded = wav2lip_faces.pad_box((0, 0, 400, 600), shape=(600, 400))
    assert padded == (0, 0, 400, 600)


def test_the_smoothing_averages_the_following_frames():
    """Upstream replaces each box with the mean of itself and the boxes
    that follow it, so a detector that jitters does not make the pasted
    face swim."""
    boxes = [(0, 0, 10, 10), (2, 0, 12, 10), (4, 0, 14, 10)]
    smoothed = wav2lip_faces.smooth_boxes(boxes, window=2)
    assert smoothed[0] == (1, 0, 11, 10)
    assert smoothed[1] == (3, 0, 13, 10)


def test_the_smoothing_at_the_end_averages_the_last_window():
    boxes = [(0, 0, 10, 10), (2, 0, 12, 10), (4, 0, 14, 10), (6, 0, 16, 10)]
    smoothed = wav2lip_faces.smooth_boxes(boxes, window=2)
    # The last box has no successor, so it averages with its predecessor.
    assert smoothed[-1] == (5, 0, 15, 10)


def test_the_smoothing_of_nothing_is_nothing():
    assert wav2lip_faces.smooth_boxes([]) == []
