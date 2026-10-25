"""Build a `lite` (FeatherTalk) avatar pack from a source video.

The `lite` model is *personalised*: the visual UNet that repaints the mouth
belongs to one person and is trained from that person's own footage. So the
build is two things, and this pipeline does both:

1. **Bake the motion source.** The video is decoded onto the 720x1280 canvas,
   a face is located in every frame, and the frames plus their face boxes are
   written into the pack. The engine replays these frames for the head and
   body motion and only the face patch is repainted.

2. **Produce the visual checkpoint.** Either a checkpoint trained elsewhere is
   imported, or the video is used to train one here (GPU, minutes to hours).

What is *not* here: any per-avatar decision the operator does not make
explicitly. A build with neither a checkpoint nor ``train`` is refused with a
message that says so -- silently training for an hour would be worse.

The training video wants to be long (three to five minutes of the person
speaking) and the pack keeps only the first stretch of it as the motion
source, so the loop that plays at runtime is short.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from server.avatar.pipeline import AvatarPipeline, PipelineError
from server.avatar import video
from server.config import get_settings
from server.lite import dataset as lite_dataset
from server.lite import frames as lite_frames
from server.lite import pack as lite_pack
from server.lite.geometry import compute_face_bbox

#: How many frames of the source video the pack keeps as the motion source.
#: The engine ping-pongs through them, so this is how long the loop is.
DEFAULT_MOTION_FRAMES = 375

#: Training hyper-parameters. The upstream defaults, kept here so the numbers
#: the operator sees are the ones the server uses.
DEFAULT_EPOCHS = 60
DEFAULT_BATCHSIZE = 8
DEFAULT_LR = 1e-3

#: How many frames without a usable face abort the build. A video where the
#: face is only occasionally visible cannot train a mouth model.
MAX_MISSING_FACE_RATIO = 0.2


class LiteAvatarPipeline(AvatarPipeline):
    """Turns an uploaded video into a `lite` pack."""

    def __init__(self, detector=None):
        # The detector is injectable so the contract test can run without the
        # SCRFD/PFLD weights.
        self._detector = detector

    # -- helpers --------------------------------------------------------

    def _get_detector(self):
        if self._detector is None:
            from server.lite.landmarks import LandmarkDetector

            self._detector = LandmarkDetector()
        return self._detector

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
        options = options or {}
        settings = get_settings()
        source_path = Path(source_path)
        out_dir = Path(out_dir)

        if not video.is_video_suffix(source_path.suffix):
            raise PipelineError(
                "轻量引擎需要一段人物口播视频（用于训练专属口型模型与头部动作），"
                "请上传视频文件而不是照片"
            )

        checkpoint = options.get("checkpoint")
        do_train = bool(options.get("train"))
        if not checkpoint and not do_train:
            raise PipelineError(
                "轻量引擎的专属权重尚不存在：请在创建形象时勾选「训练专属模型」，"
                "或提供已训练好的 checkpoint 路径"
            )

        motion_frames = int(options.get("frames") or DEFAULT_MOTION_FRAMES)
        motion_frames = max(1, min(motion_frames, settings.avatar_max_video_frames))

        work = out_dir / "lite_work"
        dataset_dir = work / "dataset"
        pack_root = lite_pack.pack_dir(out_dir)
        if pack_root.exists():
            shutil.rmtree(pack_root)
        pack_root.mkdir(parents=True, exist_ok=True)
        lite_pack.frames_dir(out_dir).mkdir(parents=True, exist_ok=True)

        try:
            kept, coords = self._bake_frames(
                source_path,
                out_dir,
                dataset_dir=dataset_dir if do_train else None,
                motion_frames=motion_frames,
                progress=progress,
            )
            if not kept:
                raise PipelineError("视频里没有检测到人脸，无法制作轻量形象")

            visual = self._visual_checkpoint(
                options,
                dataset_dir=dataset_dir,
                work=work,
                source_path=source_path,
                kept=kept,
                progress=progress,
            )
            shutil.copyfile(visual, lite_pack.visual_path(out_dir))

            lite_pack.write_coords(out_dir, coords)
            meta = {
                "version": lite_pack.VERSION,
                "frame_count": len(coords),
                "fps": settings.fps,
                "canvas": [settings.frame_width, settings.frame_height],
                "motion_frames": len(coords),
                "causal_audio": True,
            }
            lite_pack.write_meta(out_dir, meta)
            self._write_normalized(out_dir, source_path)

            progress("build_pack", 100)
            return lite_pack.config_block(
                out_dir,
                frame_count=len(coords),
                fps=settings.fps,
                model_type="feathertalk",
                base_image="normalized.png",
                causal_audio=True,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # -- frames ---------------------------------------------------------

    def _bake_frames(
        self,
        source_path: Path,
        out_dir: Path,
        *,
        dataset_dir: Path | None,
        motion_frames: int,
        progress: Callable[[str, int], None],
    ) -> tuple[list[int], list[dict]]:
        """Decode, locate the face, and write the frames.

        Returns the kept frame indices (into the decoded sequence) and the
        face box of each kept frame. When ``dataset_dir`` is given, every
        kept frame is also written there in the upstream training layout --
        the same pixels the engine will serve, so training and serving cannot
        drift apart.
        """
        settings = get_settings()
        detector = self._get_detector()
        canvas = self._canvas()

        if dataset_dir is not None:
            (dataset_dir / "full_body_img").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "landmarks").mkdir(parents=True, exist_ok=True)

        kept: list[int] = []
        coords: list[dict] = []
        total = 0
        missing = 0
        progress("detect_face", 5)
        for index, frame in lite_frames.stream_frames(
            source_path, settings.fps, canvas, max_frames=None
        ):
            total += 1
            points = detector.detect(frame)
            if points is None:
                missing += 1
                continue
            try:
                bbox = compute_face_bbox(points)
            except ValueError:
                missing += 1
                continue
            if dataset_dir is not None:
                lite_frames.write_jpeg(
                    dataset_dir / "full_body_img" / f"{len(kept)}.jpg", frame
                )
                lite_dataset.write_landmarks(
                    dataset_dir / "landmarks" / f"{len(kept)}.lms", points
                )
            if len(coords) < motion_frames:
                lite_frames.write_jpeg(lite_pack.frame_path(out_dir, len(coords)), frame)
                coords.append(
                    {
                        "bbox": [int(v) for v in bbox],
                        "size": [int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])],
                    }
                )
            kept.append(index)
            if total % 25 == 0:
                progress("detect_face", 5 + int(45 * min(1.0, len(kept) / max(1, total))))
            if dataset_dir is None and len(coords) >= motion_frames:
                break

        if total and missing > total * MAX_MISSING_FACE_RATIO:
            raise PipelineError(
                f"视频中有 {missing}/{total} 帧没有检测到人脸，素材不适合制作轻量形象"
            )
        return kept, coords

    # -- the visual checkpoint ------------------------------------------

    def _visual_checkpoint(
        self,
        options: dict,
        *,
        dataset_dir: Path | None,
        work: Path,
        source_path: Path,
        kept: list[int],
        progress: Callable[[str, int], None],
    ) -> Path:
        """Import a trained checkpoint, or train one from the video."""
        checkpoint = options.get("checkpoint")
        if checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                raise PipelineError(f"checkpoint 不存在：{path}")
            return path

        if dataset_dir is None:
            raise PipelineError("训练需要素材目录，但构建流程没有准备它")

        progress("extract_audio", 50)
        self._write_audio_features(source_path, dataset_dir, kept)

        epochs = int(options.get("epochs") or DEFAULT_EPOCHS)
        batchsize = int(options.get("batchsize") or DEFAULT_BATCHSIZE)
        lr = float(options.get("lr") or DEFAULT_LR)
        progress("train", 55)
        trained = lite_dataset.train_visual_model(
            dataset_dir,
            work / "checkpoint",
            epochs=epochs,
            batchsize=batchsize,
            lr=lr,
            progress=progress,
        )
        return _slim_checkpoint(trained, work / "visual.pth")

    def _write_audio_features(
        self, source_path: Path, dataset_dir: Path, kept: list[int]
    ) -> None:
        """Write ``aud_hu.npy``: one row of audio features per kept frame.

        The rows are re-indexed to the kept frames, because the upstream
        dataset looks up ``aud_hu[i]`` for frame ``i`` of the dataset.
        """
        features = self._audio_features(source_path)
        if features.shape[0] < len(kept):
            pad = np.zeros((len(kept) - features.shape[0],) + features.shape[1:], dtype=np.float32)
            features = np.concatenate((features, pad), axis=0)
        selected = features[np.asarray(kept[: features.shape[0]], dtype=np.int64)]
        np.save(dataset_dir / "aud_hu.npy", selected.astype(np.float32))

    def _audio_features(self, source_path: Path) -> np.ndarray:
        """The whole video's audio as ``[frames, 2, 1024]`` features."""
        from server.lite import vendor

        speech = _decode_audio_16k(source_path)
        if speech.size == 0:
            raise PipelineError("视频没有音轨，无法提取口型驱动所需的音频特征")

        import torch

        vendor.ensure_on_path()
        from feather_hubert import (
            get_feather_hubert_from_16k_speech,
            load_feather_hubert,
            make_even_first_dim,
        )

        model = load_feather_hubert(
            str(vendor.weight_path(vendor.FEATHER_HUBERT)), device=torch.device("cpu")
        )
        hidden = get_feather_hubert_from_16k_speech(speech, model, device=torch.device("cpu"))
        # The extractor hands back a tensor; the pack stores numpy.
        hidden = hidden.cpu().numpy() if hasattr(hidden, "cpu") else np.asarray(hidden)
        even = make_even_first_dim(hidden)
        return even.reshape(-1, 2, 1024).astype(np.float32)

    # -- preview --------------------------------------------------------

    def _write_normalized(self, out_dir: Path, source_path: Path) -> None:
        """The poster frame: the first frame of the motion source.

        It is what the console's preview and the session's resting canvas
        show, so it has to exist even when the engine cannot serve frames.
        """
        from PIL import Image

        path = lite_pack.frames_dir(out_dir) / "00000000.jpg"
        if path.exists():
            Image.open(path).convert("RGB").save(out_dir / "normalized.png")
            return
        # No motion frames were written (an empty pack): fall back to the
        # first decoded frame of the source.
        for _, frame in lite_frames.stream_frames(
            source_path, get_settings().fps, self._canvas(), max_frames=1
        ):
            Image.fromarray(frame[:, :, ::-1]).save(out_dir / "normalized.png")
            return


def _slim_checkpoint(trained: Path, target: Path) -> Path:
    """Keep only what the engine needs from a training checkpoint.

    The upstream saves the optimiser state next to the weights, which is
    three times the size of the model itself (66 MB against 22 MB) and is
    only useful for resuming a training run. The pack carries the model.
    """
    import torch

    checkpoint = torch.load(trained, map_location="cpu", weights_only=False)
    slim = {
        "model": checkpoint["model"],
        "model_config": checkpoint.get("model_config", {}),
        "epoch": checkpoint.get("epoch"),
    }
    torch.save(slim, target)
    return target


def _decode_audio_16k(path: Path) -> np.ndarray:
    """The video's audio as 16 kHz mono float32.

    PyAV does the decoding and the resampling; the server already depends on
    it through aiortc, and it is the only decoder here that handles every
    container the upload endpoint accepts.
    """
    """The video's audio as 16 kHz mono float32.

    PyAV does the decoding and the resampling; the server already depends on
    it through aiortc, and it is the only decoder here that handles every
    container the upload endpoint accepts.
    """
    import av

    container = av.open(str(path))
    try:
        streams = container.streams.audio
        if not streams:
            return np.zeros(0, dtype=np.float32)
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=16000
        )
        chunks = []
        for frame in container.decode(streams[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio
    finally:
        container.close()
