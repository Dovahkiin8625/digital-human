"""Face detection for the Wav2Lip model.

The upstream crop geometry is defined against **S3FD**'s boxes: the network
was trained on faces cut where S3FD found them, so the detector is part of
the model rather than an interchangeable accessory. This module drives the
S3FD detector from the vendored checkout and applies the two adjustments
upstream makes to its boxes:

* a **bottom pad** of 10 pixels, because the crop has to include the chin --
  the mouth is at the bottom of the face, and a crop that cuts it off leaves
  the model nothing to paint;
* a **smoothing** pass over five frames, because a detector that jitters by a
  pixel from frame to frame makes the pasted face swim.

Both are pure functions of the boxes, so they are testable without torch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- typing only
    Bbox = tuple[int, int, int, int]

#: Upstream's `--pads` default: (top, bottom, left, right).
PADS = (0, 10, 0, 0)

#: Upstream smooths each box over this many frames.
SMOOTH_WINDOW = 5

#: How many frames are handed to the detector at once. The detector runs one
#: forward pass per batch, so a bigger batch is faster but holds more memory.
DETECT_BATCH = 16


def pad_box(box: "Bbox", shape: tuple[int, int], pads=PADS) -> "Bbox":
    """Upstream's padding, clipped to the frame.

    ``shape`` is the frame's ``(height, width)``.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    top, bottom, left, right = pads
    height, width = shape
    return (
        max(0, x1 - left),
        max(0, y1 - top),
        min(width, x2 + right),
        min(height, y2 + bottom),
    )


def smooth_boxes(boxes: list["Bbox"], window: int = SMOOTH_WINDOW) -> list["Bbox"]:
    """Upstream's `get_smoothened_boxes`: the mean over a forward window.

    Each box is replaced by the mean of itself and the boxes that follow it;
    the last few boxes average the final window instead. Upstream does this
    in place over the whole video, which is why it is done at build time here
    and the smoothed boxes are what the pack stores.
    """
    if not boxes:
        return []
    array = np.asarray(boxes, dtype=np.float64)
    out: list["Bbox"] = []
    for i in range(len(boxes)):
        if i + window > len(boxes):
            chunk = array[len(boxes) - window :]
        else:
            chunk = array[i : i + window]
        mean = chunk.mean(axis=0)
        out.append((int(mean[0]), int(mean[1]), int(mean[2]), int(mean[3])))
    return out


class FaceDetector:
    """The vendored S3FD detector, wrapped.

    The detector is created once and reused: loading S3FD costs a second, and
    a build runs it over every frame of the source video.
    """

    def __init__(self, weights_path, device: str = "cpu"):
        from dh_server.wav2lip import vendor

        module = vendor.import_vendor("face_detection.detection.sfd.sfd_detector")
        self._detector = module.SFDDetector(device=device, path_to_detector=str(weights_path))

    def detect(self, frames: list[np.ndarray]) -> list["Bbox" | None]:
        """The face box of every frame, or ``None`` where no face was found.

        ``frames`` are the canvas frames in BGR -- upstream's own frames are
        BGR too, and the detector's colour convention is part of what the
        model was trained against.
        """
        import torch

        boxes: list["Bbox" | None] = []
        # The vendored detector switches cuDNN's benchmarking on, globally,
        # the first time it runs on a GPU. Left on, every *new* input shape
        # any model in this process sees afterwards costs an exhaustive
        # algorithm search -- seconds per shape, and the first session after
        # a build is the worst place for that. The detector's own shapes are
        # constant, so it only needs the setting for its own run.
        benchmark = torch.backends.cudnn.benchmark
        try:
            for start in range(0, len(frames), DETECT_BATCH):
                batch = frames[start : start + DETECT_BATCH]
                # The detector subtracts the OpenCV means, so it wants RGB --
                # upstream's own API flips the frames before handing them over.
                images = np.asarray(batch, dtype=np.uint8)[..., ::-1]
                found = self._detector.detect_from_batch(images)
                for item in found:
                    boxes.append(self._first_box(item, batch[0].shape[:2]))
        finally:
            torch.backends.cudnn.benchmark = benchmark
        return boxes

    @staticmethod
    def _first_box(found, shape) -> "Bbox" | None:
        """The first box of one frame, or ``None``.

        Upstream takes the first box the detector returns for a frame, so this
        does the same rather than picking by score.
        """
        if found is None or len(found) == 0:
            return None
        best = found[0]
        x1, y1, x2, y2 = (int(v) for v in np.clip(best[:-1], 0, None))
        if x2 <= x1 or y2 <= y1:
            return None
        return pad_box((x1, y1, x2, y2), shape)
