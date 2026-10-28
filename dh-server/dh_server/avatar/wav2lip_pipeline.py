"""Build a `wav2lip` avatar pack from a source video or photo.

Wav2Lip is a *generic* model: one checkpoint drives every avatar, so the
build has no training step and no per-avatar weights. What it does have is a
face to drive, and a face to drive it with:

1. **Bake the motion source.** The source is decoded onto the 720x1280
   canvas, a face is located in every frame, and the frames plus their face
   boxes are written into the pack. The engine replays these frames for the
   head and body motion and only the face is repainted.

2. **Paint the idle track.** The motion source is a video of the person
   *speaking*, so playing it back while the session is silent would show the
   avatar talking to nobody. The same frames are therefore repainted with a
   silent mel window, which closes the mouth (see :mod:`dh_server.wav2lip.idle`).

What is *not* here: any per-avatar decision the operator does not make
explicitly. A source with no detectable face fails the build rather than
producing an avatar that cannot be driven.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from dh_server.avatar import video
from dh_server.avatar.pipeline import AvatarPipeline, PipelineError
from dh_server.config import get_settings
from dh_server.wav2lip import faces as wav2lip_faces
from dh_server.wav2lip import frames as wav2lip_frames
from dh_server.wav2lip import pack as wav2lip_pack

#: How many frames with a usable face a source must yield. Below this the
#: motion source is too short to loop without the avatar visibly repeating.
MIN_FRAMES = 8


class Wav2LipAvatarPipeline(AvatarPipeline):
    """Turns an uploaded video or photo into a `wav2lip` pack."""

    def __init__(self, detector=None, model=None):
        # The detector and the model are injectable so the contract test can
        # run without the S3FD weights or the checkpoint.
        self._detector = detector
        self._model = model

    # -- helpers --------------------------------------------------------

    def _get_detector(self):
        if self._detector is None:
            from dh_server.wav2lip import vendor
            from dh_server.wav2lip.model import pick_device

            self._detector = wav2lip_faces.FaceDetector(
                vendor.weight_path(vendor.S3FD), device=pick_device()
            )
        return self._detector

    def _get_model(self):
        if self._model is None:
            from dh_server.wav2lip import vendor
            from dh_server.wav2lip.model import Wav2LipModel, pick_device

            self._model = Wav2LipModel(
                vendor.weight_path(vendor.GENERATOR), device=pick_device()
            )
        return self._model

    def _canvas(self) -> tuple[int, int]:
        settings = get_settings()
        return settings.frame_width, settings.frame_height

    # -- the build ------------------------------------------------------

    def process(
        self,
        avatar_id: str,
        source_path: Path,
        out_dir: Path,
        progress: Callable[[str, int], None],
        *,
        options: dict | None = None,
    ) -> dict:
        # The model needs no build options: Wav2Lip is generic, so there is
        # nothing to train and no checkpoint to import.
        settings = get_settings()
        source_path = Path(source_path)
        out_dir = Path(out_dir)

        pack_root = wav2lip_pack.pack_dir(out_dir)
        if pack_root.exists():
            shutil.rmtree(pack_root)
        pack_root.mkdir(parents=True, exist_ok=True)
        wav2lip_pack.frames_dir(out_dir).mkdir(parents=True, exist_ok=True)

        built = False
        try:
            frames, is_video = self._source_frames(source_path)
            progress("detect_face", 20)
            boxes = self._detect(frames)
            kept = [i for i, box in enumerate(boxes) if box is not None]
            # A photo is the one-frame case and is allowed: Wav2Lip paints the
            # mouth onto a still face. A video has to carry enough motion to
            # loop without the avatar visibly repeating.
            needed = MIN_FRAMES if is_video else 1
            if len(kept) < needed:
                raise PipelineError(
                    f"素材里只找到 {len(kept)} 张有人脸的画面，至少需要 {needed} 张；"
                    "请换一段人脸清晰、不遮挡的视频"
                )
            # The boxes are smoothed over the frames that survived: a detector
            # that jitters by a pixel makes the pasted face swim.
            smoothed = wav2lip_faces.smooth_boxes([boxes[i] for i in kept])

            coords: list[dict] = []
            for index, box in zip(kept, smoothed):
                wav2lip_frames.write_jpeg(
                    wav2lip_pack.frame_path(out_dir, len(coords)), frames[index]
                )
                coords.append({"bbox": [int(v) for v in box]})
            wav2lip_pack.write_coords(out_dir, coords)

            progress("idle_track", 70)
            idle_count = self._bake_idle_track(out_dir, coords, progress)

            meta = {
                "version": wav2lip_pack.VERSION,
                "frame_count": len(coords),
                "fps": settings.fps,
                "canvas": [settings.frame_width, settings.frame_height],
                "idle_frames": idle_count,
            }
            wav2lip_pack.write_meta(out_dir, meta)
            self._write_normalized(out_dir)

            progress("build_pack", 100)
            built = True
            return wav2lip_pack.config_block(
                frame_count=len(coords),
                fps=settings.fps,
                model_type="wav2lip",
                base_image="normalized.png",
                idle_frames=idle_count > 0,
            )
        finally:
            if not built:
                # A half-built pack is worse than none: the job failed, and a
                # retry rebuilds from scratch anyway.
                shutil.rmtree(pack_root, ignore_errors=True)

    # -- the source -----------------------------------------------------

    def _source_frames(self, source_path: Path) -> tuple[list[np.ndarray], bool]:
        """The source's frames on the canvas, in BGR, and whether it was a video."""
        settings = get_settings()
        if video.is_video_suffix(source_path.suffix):
            frames = [
                frame
                for _, frame in wav2lip_frames.stream_frames(
                    source_path,
                    settings.fps,
                    self._canvas(),
                    max_frames=settings.avatar_max_video_frames,
                )
            ]
            if not frames:
                raise PipelineError("视频里没有可解码的画面")
            return frames, True
        return [self._image_frame(source_path)], False

    def _image_frame(self, source_path: Path) -> np.ndarray:
        """A photo, letterboxed onto the canvas, in BGR."""
        from PIL import Image

        settings = get_settings()
        try:
            image = Image.open(source_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 -- the upload validator's business
            raise PipelineError(f"图片无法解码：{exc}") from exc
        canvas = video.fit_to_canvas(image, *self._canvas())
        return np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])

    # -- detection ------------------------------------------------------

    def _detect(self, frames: list[np.ndarray]) -> list[tuple | None]:
        """The face box of every frame, or ``None`` where there is no face."""
        return self._get_detector().detect(frames)

    # -- the idle track -------------------------------------------------

    def _bake_idle_track(self, out_dir: Path, coords: list[dict], progress) -> int:
        """Paint the closed-mouth idle track (see :mod:`dh_server.wav2lip.idle`).

        A failure here fails the build: a pack without the track ships an
        avatar that talks while the session is silent.
        """
        from dh_server.wav2lip import idle as wav2lip_idle

        model = self._get_model()
        return wav2lip_idle.bake_idle_frames(out_dir, coords, model)

    # -- preview --------------------------------------------------------

    def _write_normalized(self, out_dir: Path) -> None:
        """The poster frame: the first idle frame, or the first motion frame.

        It is what the console's preview and the session's resting canvas
        show, so it has to exist even when the engine cannot serve frames.
        The idle frame is preferred: its mouth is closed, which is what a
        resting avatar should look like.
        """
        from PIL import Image

        for path in (
            wav2lip_pack.idle_frame_path(out_dir, 0),
            wav2lip_pack.frame_path(out_dir, 0),
        ):
            if path.exists():
                Image.open(path).convert("RGB").save(out_dir / "normalized.png")
                return
