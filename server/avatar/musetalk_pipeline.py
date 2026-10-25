"""Real MuseTalk 1.5 avatar preparation -- the production path.

Turns one uploaded photo or one short driver video into the artifacts the engine
needs. A photo is the N=1 case of a frame cycle, so both modes write the same
layout:

    <avatar_dir>/normalized.png          720x1280 base frame, taken from the first
                                         kept frame (idle keepalive, preview, and
                                         the shape the old single-frame packs had)
    <avatar_dir>/pack.json               small scalars/paths only (no tensors)
    <avatar_dir>/musetalk/
        latents.pt      [N,8,32,32]      VAE latents, one row per kept frame
        coords.json     [N]              face box + blend crop box per kept frame
        masks/NNNNNNNN.png               bisenet blend mask per kept frame, sized
                                         to that frame's own mask_crop_box
        meta.json       model version / crop / detector / fps / mode / kept

`kept` is the list of decoded-sequence indices the N rows came from: a video
drops frames with no detectable face, so row i of `coords.json` is decoded frame
`kept[i]`, and the engine re-decodes the same sequence and selects those indices.

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
from server.avatar.quality import assess as assess_quality
from server.avatar.video import is_video_suffix
from server.config import get_settings

SUBDIR = "musetalk"
FACE_CROP = 256
VERSION = "1.5"
DETECTOR = "mediapipe.face_landmarker"
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "MuseTalk"

#: A video needs at least this many face-bearing frames to be usable as a driver.
MIN_FRAMES = 8

#: Serializes the first load of MuseTalk's vendored models, which pull in
#: `diffusers`. Two threads reach that import in normal operation -- the
#: engine's warmup on its worker thread, and an avatar job on the job thread --
#: and `diffusers/__init__.py` ends by replacing itself in `sys.modules` with a
#: lazy module. A second thread resolving a name during that window sees a
#: module without it. Reproduced: uploading an avatar while warmup was still
#: loading failed with "cannot import name 'AutoencoderKL' from 'diffusers'";
#: the same upload after warmup succeeded. Held across construction, not just
#: the import, so the two never interleave their ~11 s of GPU init either.
IMPORT_LOCK = threading.Lock()

#: The files `_Runtime` (the engine's global model set) loads: the MuseTalk 1.5
#: UNet and its config, the SD-VAE (config + weights) and whisper-tiny (config,
#: weights and feature-extractor config). Shared with the engine so one list
#: decides "are the weights present" for both `describe()`/`warmup()` and this
#: pipeline's own check. When the two lists drifted, describe() reported
#: weights_present: True and then failed inside `_Runtime` on a file the list
#: never mentioned -- the point of the check is to fail before that.
RUNTIME_WEIGHTS = (
    "musetalkV15/unet.pth",
    "musetalkV15/musetalk.json",
    "sd-vae/config.json",
    "sd-vae/diffusion_pytorch_model.bin",
    "whisper/config.json",
    "whisper/pytorch_model.bin",
    "whisper/preprocessor_config.json",
)


def _cv2_and_torch():
    """Import the GPU-venv-only deps at call time.

    This module must import cleanly in the torch-free `.venv` so that a
    misconfigured environment reports a clear error instead of an
    ImportError at import time.
    """
    import cv2
    import torch

    return cv2, torch


def _save_latents(parts, path: Path) -> None:
    """Seam for the torch-free contract test.

    `parts` is a list of per-frame ``[1,8,32,32]`` latents; they are stacked to
    ``[N,8,32,32]`` and saved as one tensor. Tests substitute a numpy writer.
    """
    _, torch = _cv2_and_torch()
    torch.save(torch.cat(list(parts), dim=0).detach().cpu(), path)


def default_models_dir() -> Path:
    return Path(get_settings().data_dir) / "models" / "musetalk"


def fit_to_canvas(img: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox `img` into a width x height canvas, preserving aspect ratio.

    The pipeline must produce a 720x1280 `normalized.png` because
    `RenderSession` assumes the media contract's frame size.
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

        with IMPORT_LOCK:
            from musetalk.models.vae import VAE
            from musetalk.utils.blending import get_image_prepare_material

            self._get_image_prepare_material = get_image_prepare_material
            self._vae = VAE(model_path=str(self.models_dir / "sd-vae"))
            self._vae.vae = self._vae.vae.half().to(self._device())
            self._fp = self._face_parsing()
            self._landmarker, self._mp = self._build_landmarker()

    # -- model plumbing -------------------------------------------------
    def _missing_weights(self) -> None:
        # The engine's runtime set, plus the files only preparation needs: the
        # biSeNet face parser and its resnet18 backbone, and the mediapipe
        # landmarker. The engine never loads those, so they must not gate
        # describe()/warmup() -- but a prepare() without them must still fail
        # loudly here.
        needed = list(RUNTIME_WEIGHTS) + [
            "face-parse-bisent/79999_iter.pth",
            "face-parse-bisent/resnet18-5c106cde.pth",
            "face_landmarker.task",
        ]
        missing = [str(self.models_dir / w) for w in needed if not (self.models_dir / w).exists()]
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
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress: Callable[[str, int], None],
        *,
        options: dict | None = None,
    ) -> dict:
        settings = get_settings()
        source_path = Path(source_path)
        # The ``mode`` field drives engine-side frame cycling and the idle
        # video endpoint. It is purely a function of the source file: video
        # uploads use ``mode="video"`` (engine cycles through the decoded
        # frames), still photos use ``mode="image"`` (engine reads one frame
        # and loops). There is no longer an "animate first" branch -- that
        # responsibility moved to the standalone "/i2v" endpoint under the
        # "形象活化" tab.
        is_video_upload = is_video_suffix(source_path.suffix)
        mode = "video" if is_video_upload else "image"

        progress("detect_face", 10)
        frames: list[np.ndarray]
        source_fps: float
        if is_video_upload:
            frames, source_fps = self._video_frames(source_path, settings)
        else:
            frames, source_fps = self._image_frames(source_path, settings)

        prep = self._get_preparer()
        parts, coords, masks, kept = [], [], [], []
        warnings: list[str] = []
        for i, bgr in enumerate(frames):
            try:
                out = prep(bgr)
            except PipelineError:
                if mode == "image":
                    raise  # an image with no face is a plain error, not a skip
                continue
            if not kept:
                # First usable frame stands for the source: one advisory
                # assessment per job, on the prepared canvas, in RGB.
                warnings = list(warnings) + list(
                    assess_quality(np.ascontiguousarray(bgr[:, :, ::-1]), out["landmarks"])
                )
            kept.append(i)
            parts.append(out["latents"])
            coords.append({"bbox": out["bbox"], "mask_crop_box": out["mask_crop_box"]})
            masks.append(out["mask"])

        if not parts and mode == "image":
            raise PipelineError("no face detected in the uploaded image")
        if mode == "video" and len(parts) < MIN_FRAMES:
            raise PipelineError(f"no usable face frames in video: found {len(parts)}, need {MIN_FRAMES}")

        progress("align", 40)
        sidecar = out_dir / SUBDIR
        sidecar.mkdir(parents=True, exist_ok=True)
        masks_dir = sidecar / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        _save_latents(parts, sidecar / "latents.pt")
        (sidecar / "coords.json").write_text(json.dumps(coords))
        for i, mask in enumerate(masks):
            Image.fromarray(mask).save(masks_dir / f"{i:08d}.png")

        progress("normalize", 70)
        # Frame 0 of the *kept* frames: the resting canvas, preview and back-compat.
        Image.fromarray(frames[kept[0]][:, :, ::-1]).save(out_dir / "normalized.png")

        progress("build_pack", 100)
        meta = {
            "version": VERSION,
            "crop": FACE_CROP,
            "detector": DETECTOR,
            "fps": settings.fps,
            "source_fps": source_fps,
            "frame_count": len(parts),
            "mode": mode,
            "kept": kept,
        }
        (sidecar / "meta.json").write_text(json.dumps(meta))
        config: dict = {
            "musetalk": {
                "mode": mode,
                "dir": SUBDIR,
                "version": VERSION,
                "crop": FACE_CROP,
                "detector": DETECTOR,
                "frame_count": len(parts),
                "fps": settings.fps,
                "source_fps": source_fps,
                "base_image": "normalized.png",
                # The source artifact is whatever the upload produced; video
                # uploads carry their own bytes, image uploads are read
                # back from the saved source_image_path. The engine uses
                # ``mode`` to pick which one to load.
                "source": source_path.name,
                "latents": f"{SUBDIR}/latents.pt",
                "coords": f"{SUBDIR}/coords.json",
                "masks": f"{SUBDIR}/masks",
                "bbox": coords[0]["bbox"],
                "mask_crop_box": coords[0]["mask_crop_box"],
            }
        }
        if warnings:
            config["warnings"] = warnings
        return config

    def _image_frames(self, source_path: Path, settings) -> tuple[list[np.ndarray], float]:
        img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(
                f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}"
            )
        canvas = fit_to_canvas(img, settings.frame_width, settings.frame_height)
        return [np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])], float(settings.fps)

    def _video_frames(self, source_path: Path, settings) -> tuple[list[np.ndarray], float]:
        from server.avatar.video import decode_frames

        frames, source_fps = decode_frames(
            source_path,
            target_fps=settings.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if frames.shape[0] == 0:
            raise PipelineError("video has no decodable frames")
        # The preparer expects BGR; decode_frames returns the canvas in RGB.
        return [np.ascontiguousarray(f[:, :, ::-1]) for f in frames], source_fps
