"""Face landmarks for the `lite` model: SCRFD detection + PFLD refinement.

The pipeline needs 110 landmarks per frame to place the face crop, and the
*runtime* needs them not at all -- the crop is baked into the pack. So this
runs once, at build time.

Upstream does the same two steps (SCRFD 2.5G KPS for the box, PFLD-MobileOne
for the 110 points) but loads its weights relative to the current working
directory, so it has to be run with ``cwd=data_utils``. This module passes
absolute paths instead: the server never `chdir`s itself. (The MuseTalk
`FaceParsing` weights taught us that lesson -- its relative paths only worked
when the process happened to be started inside `vendor/MuseTalk`.)

Torch is imported lazily; `cv2` too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from server.lite import vendor

#: The number of landmarks PFLD-MobileOne emits.
LANDMARK_COUNT = 110
#: The square the detected face is cropped to before PFLD sees it.
PFLD_INPUT = 192
#: The face box is padded by this factor before cropping.
BOX_PADDING = 1.05


class LandmarkDetector:
    """Detects one face per frame and returns its 110 landmarks."""

    def __init__(self, weights_dir: Path | None = None, device: str = "cpu"):
        import torch

        vendor.ensure_on_path()
        from base_module import MobileOneBlock  # noqa: F401 -- path setup
        from detect_face import SCRFD
        from pfld_mobileone import PFLD_GhostOne

        weights = Path(weights_dir) if weights_dir else vendor.weights_dir()
        scrfd_path = weights / vendor.SCRFD
        pfld_path = weights / vendor.PFLD
        for path in (scrfd_path, pfld_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing; run scripts/fetch_lite_weights.sh"
                )

        mean_face = vendor.vendor_dir() / "data_utils" / "mean_face.txt"
        self._mean_face = np.asarray(mean_face.read_text().split(" "), dtype=np.float32)

        self._device = torch.device(device)
        self._detector = SCRFD(str(scrfd_path), confThreshold=0.1, nmsThreshold=0.5)
        checkpoint = torch.load(str(pfld_path), map_location=self._device)
        self._backbone = PFLD_GhostOne().to(self._device)
        self._backbone.load_state_dict(checkpoint["pfld_backbone"])
        self._backbone.eval()

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """The 110 landmarks of the largest face in ``frame``.

        ``frame`` is HWC BGR (OpenCV's order). The result is ``[110, 2]`` in
        the frame's pixel coordinates, or ``None`` when no face was found.
        """
        import cv2
        import torch

        crop, origin = self._crop_face(frame)
        if crop is None:
            return None
        height, width = crop.shape[:2]
        resized = cv2.resize(crop, (PFLD_INPUT, PFLD_INPUT))
        tensor = torch.from_numpy(
            (np.asarray(resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        )[None].to(self._device)
        with torch.no_grad():
            landmarks = self._backbone(tensor)
        points = landmarks[0].cpu().numpy() + self._mean_face
        points = points.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 0] += origin[0]
        points[:, 1] += origin[1]
        return points.astype(np.int32)

    def _crop_face(self, frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int]]:
        """The square face crop and its origin, or ``(None, (0, 0))``."""
        import cv2

        boxes, indices, _ = self._detector.detect(frame)
        if not len(indices):
            return None, (0, 0)
        index = indices[0]
        x1 = int(boxes[index, 0])
        y1 = int(boxes[index, 1])
        x2 = int(boxes[index, 0] + boxes[index, 2])
        y2 = int(boxes[index, 1] + boxes[index, 3])
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None, (0, 0)

        size = int(max(width, height) * BOX_PADDING)
        cx = (x2 + x1) // 2
        cy = (y2 + y1) // 2
        left = cx - size // 2
        top = cy - size // 2

        frame_height, frame_width = frame.shape[:2]
        dx = max(0, -left)
        dy = max(0, -top)
        left = max(0, left)
        top = max(0, top)
        right = min(frame_width, left + size)
        bottom = min(frame_height, top + size)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None, (0, 0)
        if dx > 0 or dy > 0 or right < left + size or bottom < top + size:
            crop = cv2.copyMakeBorder(
                crop,
                dy,
                max(0, top + size - bottom),
                dx,
                max(0, left + size - right),
                cv2.BORDER_CONSTANT,
                0,
            )
            top -= dy
            left -= dx
        return crop, (left, top)
