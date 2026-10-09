"""Real MuseTalk 1.5 avatar preparation -- the production path.

Turns one uploaded photo into the artifacts the engine needs, and nothing more:

    <avatar_dir>/normalized.png          720x1280 base frame (detection runs here,
                                         so every stored coordinate is already in
                                         canvas space -- the engine never rescales)
    <avatar_dir>/pack.json               small scalars/paths only (no tensors)
    <avatar_dir>/musetalk/
        face_crop.png   256x256          aligned face, for inspection/debugging
        bbox.json       face box + blend crop box
        landmarks.json  detector output
        mask.png        bisenet blend mask (sized to the blend crop box)
        latents.pt      [1,8,32,32] VAE latents
        meta.json       model version / crop / detector

Everything torch-shaped is imported lazily inside `MuseTalkPreparer`, so this
module stays importable (and the default test suite stays torch-free) even
though it is only ever *used* in the GPU venv.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from server.avatar.landmarks import CHIN, LEFT_CHEEK, NOSE_TIP, RIGHT_CHEEK
from server.avatar.pipeline import AvatarPipeline, PipelineError
from server.config import get_settings

SUBDIR = "musetalk"
FACE_CROP = 256
VERSION = "1.5"
DETECTOR = "mediapipe.face_landmarker"
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "MuseTalk"


def _cv2_and_torch():
    """Import the GPU-venv-only deps at call time.

    This module must import cleanly in the torch-free `.venv` so that a
    misconfigured DH_AVATAR_PIPELINE reports a clear error instead of an
    ImportError at import time.
    """
    import cv2
    import torch

    return cv2, torch


def _save_latents(latents, path: Path) -> None:
    """Seam for the torch-free contract test.

    Serialising a `.pt` needs torch, which only exists in the GPU venv, but the
    rest of the pack contract (file names, coordinate propagation, pack.json
    shape) is exactly what a GPU-less test can and should assert. Tests
    substitute this with a numpy writer.
    """
    _, torch = _cv2_and_torch()
    torch.save(latents.detach().cpu(), path)


def default_models_dir() -> Path:
    return Path(get_settings().data_dir) / "models" / "musetalk"


def fit_to_canvas(img: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox `img` into a width x height canvas, preserving aspect ratio.

    The mock pipeline already guarantees a 720x1280 `normalized.png`; the real
    one must produce the same shape because `RenderSession` assumes the media
    contract's frame size.
    """
    canvas = Image.new("RGB", (width, height), (16, 16, 24))
    scale = min(width / img.width, height / img.height)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resized = img.resize(new, Image.LANCZOS)
    canvas.paste(resized, ((width - new[0]) // 2, (height - new[1]) // 2))
    return canvas


class MuseTalkPreparer:
    """Owns the torch-side models and converts one BGR canvas into pack artifacts.

    Loading costs ~11 s and the first inference compiles CUDA kernels for
    another ~10 s, so instances are cached per models_dir and shared across
    avatar jobs.
    """

    def __init__(self, models_dir: Path):
        import sys

        import cv2
        import torch

        self._cv2 = cv2
        self._torch = torch
        self.models_dir = Path(models_dir)
        self._missing_weights()

        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))

        from musetalk.models.vae import VAE
        from musetalk.utils.blending import get_image_prepare_material

        self._get_image_prepare_material = get_image_prepare_material
        self._vae = VAE(model_path=str(self.models_dir / "sd-vae"))
        self._vae.vae = self._vae.vae.half().to(self._device())
        self._fp = self._face_parsing()
        self._landmarker, self._mp = self._build_landmarker()

    # -- model plumbing -------------------------------------------------
    def _missing_weights(self) -> None:
        needed = [
            self.models_dir / "musetalkV15" / "unet.pth",
            self.models_dir / "sd-vae" / "config.json",
            self.models_dir / "face-parse-bisent" / "79999_iter.pth",
            self.models_dir / "face-parse-bisent" / "resnet18-5c106cde.pth",
            self.models_dir / "face_landmarker.task",
        ]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise PipelineError(
                f"musetalk weights not found in {self.models_dir} (missing: {missing}); "
                "run scripts/fetch_musetalk_weights.sh"
            )

    def _device(self):
        torch = self._torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _face_parsing(self):
        """`FaceParsing` hardcodes './models/face-parse-bisent/*.pth' defaults and
        calls `model_init()` with no arguments, so it only works with CWD ==
        vendor/MuseTalk. Subclass to pass absolute paths instead of chdir'ing the
        whole server process.
        """
        from musetalk.utils.face_parsing import FaceParsing

        weights = self.models_dir / "face-parse-bisent"

        class _AbsPathFaceParsing(FaceParsing):
            def model_init(self, resnet_path=None, model_pth=None):
                return super().model_init(
                    resnet_path or str(weights / "resnet18-5c106cde.pth"),
                    model_pth or str(weights / "79999_iter.pth"),
                )

        return _AbsPathFaceParsing(left_cheek_width=90, right_cheek_width=90)

    def _build_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.models_dir / "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(opts), mp

    # -- detection ------------------------------------------------------
    def landmarks(self, bgr: np.ndarray) -> np.ndarray:
        """(N,2) pixel landmarks in canvas space, or raise if no face found."""
        mp = self._mp
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            raise PipelineError("no face detected in the uploaded image")
        lm = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    @staticmethod
    def crop_box(pts: np.ndarray, frame_hw: tuple[int, int], extra_margin: int = 10) -> list[int]:
        """MuseTalk's landmark-derived crop (v1.5, bbox_shift=0).

        Mirrors `musetalk.utils.preprocessing.get_landmark_and_bbox`, but from
        mediapipe landmarks rather than DWPose:

            nose_y = pts[NOSE_TIP].y ; chin_y = pts[CHIN].y
            upper  = max(0, nose_y - (chin_y - nose_y))
            box    = (min_x, upper, max_x, min(chin_y + margin, H))
        """
        nose_y = float(pts[NOSE_TIP, 1])
        chin_y = float(pts[CHIN, 1])
        upper = max(0.0, nose_y - (chin_y - nose_y))
        x1 = float(min(pts[LEFT_CHEEK, 0], pts[RIGHT_CHEEK, 0]))
        x2 = float(max(pts[LEFT_CHEEK, 0], pts[RIGHT_CHEEK, 0]))
        y2 = min(chin_y + extra_margin, frame_hw[0])
        box = [int(round(x1)), int(round(upper)), int(round(x2)), int(round(y2))]
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            raise PipelineError(f"degenerate face crop {box}")
        return box

    # -- entry point ----------------------------------------------------
    def __call__(self, bgr: np.ndarray) -> dict:
        cv2 = self._cv2
        torch = self._torch
        pts = self.landmarks(bgr)
        bbox = self.crop_box(pts, bgr.shape[:2])
        x1, y1, x2, y2 = bbox
        resized = cv2.resize(
            bgr[y1:y2, x1:x2], (FACE_CROP, FACE_CROP), interpolation=cv2.INTER_LANCZOS4
        )
        with torch.no_grad():
            latents = self._vae.get_latents_for_unet(resized)  # [1,8,32,32]
        mask, mask_crop_box = self._get_image_prepare_material(bgr, bbox, fp=self._fp, mode="jaw")
        return {
            "bbox": bbox,
            "landmarks": pts,
            "face_crop_bgr": resized,
            "latents": latents,
            "mask": mask,
            "mask_crop_box": [int(v) for v in mask_crop_box],
        }


_shared: dict[str, MuseTalkPreparer] = {}
_shared_lock = threading.Lock()


def get_preparer(models_dir: Path) -> MuseTalkPreparer:
    key = str(Path(models_dir).resolve())
    with _shared_lock:
        if key not in _shared:
            _shared[key] = MuseTalkPreparer(models_dir)
        return _shared[key]


class MuseTalkAvatarPipeline(AvatarPipeline):
    MIN_SIZE = 256

    def __init__(self, models_dir: Path | None = None, preparer=None):
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        # Injectable so the contract can be asserted without torch or a GPU.
        self._preparer = preparer

    def _get_preparer(self):
        if self._preparer is None:
            self._preparer = get_preparer(self.models_dir)
        return self._preparer

    def process(
        self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]
    ) -> dict:
        settings = get_settings()

        progress("detect_face", 10)
        img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(
                f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}"
            )
        canvas = fit_to_canvas(img, settings.frame_width, settings.frame_height)
        bgr = np.asarray(canvas)[:, :, ::-1].copy()

        prep = self._get_preparer()(bgr)

        progress("align", 40)
        sidecar = out_dir / SUBDIR
        sidecar.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prep["face_crop_bgr"][:, :, ::-1]).save(sidecar / "face_crop.png")
        (sidecar / "bbox.json").write_text(
            json.dumps({"bbox": prep["bbox"], "mask_crop_box": prep["mask_crop_box"]})
        )
        (sidecar / "landmarks.json").write_text(
            json.dumps([[round(float(x), 2), round(float(y), 2)] for x, y in prep["landmarks"]])
        )

        progress("normalize", 70)
        canvas.save(out_dir / "normalized.png")
        # PIL rather than cv2 here so that nothing in this method needs the GPU venv.
        Image.fromarray(prep["mask"]).save(sidecar / "mask.png")
        _save_latents(prep["latents"], sidecar / "latents.pt")

        progress("build_pack", 100)
        (sidecar / "meta.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "crop": FACE_CROP,
                    "detector": DETECTOR,
                    "latent_shape": list(prep["latents"].shape),
                    "fps": settings.fps,
                    "sample_rate": settings.sample_rate,
                }
            )
        )
        return {
            "musetalk": {
                "dir": SUBDIR,
                "version": VERSION,
                "crop": FACE_CROP,
                "detector": DETECTOR,
                "bbox": prep["bbox"],
                "mask_crop_box": prep["mask_crop_box"],
                "base_image": "normalized.png",
                "latents": f"{SUBDIR}/latents.pt",
                "mask": f"{SUBDIR}/mask.png",
            }
        }
