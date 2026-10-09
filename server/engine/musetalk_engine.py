"""Real MuseTalk 1.5 drive engine.

Turns a prepared avatar pack plus PCM into lip-synced 720x1280 frames using the
vendored MuseTalk 1.5 models (VAE, UNet, whisper-tiny). No OpenMMLab: the only
mm* user in MuseTalk is DWPose, and the preparation pipeline already replaced it
with mediapipe, so nothing here imports mmcv/mmdet/mmpose/mmengine.

Model state is layered:

  * global weights   -- VAE/UNet/whisper, loaded once, lock-protected, warmed
                        at startup (cold start is ~34 s of loading + kernel
                        autotune, which must not land on the first session)
  * per-avatar cache -- latents/mask/base frame, LRU by avatar_id

Every CUDA call runs on a single-worker ThreadPoolExecutor. Blocking inference
therefore never stalls the event loop, and concurrent sessions queue on the one
GPU instead of interleaving kernels -- the same shape as MuseTalk's own single
`process_frames` thread.

Torch is imported lazily inside methods, so this module still imports in the
torch-free `.venv`; the default suite constructs it with stubbed internals.
"""
from __future__ import annotations

import asyncio
import functools
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.avatar.musetalk_pipeline import VENDOR_DIR
from server.config import get_settings
from server.engine.base import (
    AvatarPack,
    DriveEngine,
    DriveFrame,
    EngineUnavailable,
    gpu_available,
)

# whisper and MuseTalk both work at 16 kHz; ingest is 48 kHz.
SR16 = 16000
# MuseTalk's whisper feature stream is 50 fps and each output frame consumes a
# 10-wide window of it (2 left + 2 right of padding, +/-1).
WHISPER_FPS = 50
# Left context, in 25 fps output frames, replayed through whisper each block so
# the prompt for the first new frame is not computed off a truncated window.
CONTEXT_FRAMES = 5
# 2 s of 16 kHz audio: far more than CONTEXT_FRAMES plus one 100 ms block, and
# re-trimmed every call so the buffer cannot grow without bound.
MAX_BUF = SR16 * 2
AVATAR_CACHE_SIZE = 4
BATCH_SIZE = 8

_decimators: dict[int, np.ndarray] = {}


def _design_decimator(ratio: int, taps: int = 33) -> np.ndarray:
    """Windowed-sinc low-pass for integer decimation by `ratio`.

    Cutoff sits at the output Nyquist (0.5/ratio of the input rate) and a Kaiser
    window keeps the stopband low enough that whisper's mel input is
    indistinguishable from scipy's resample_poly for speech -- at the cost of a
    scipy dependency in the base venv, which the default suite does not have.
    """
    n = np.arange(taps) - (taps - 1) / 2
    cutoff = 0.5 / ratio
    h = np.sinc(2 * cutoff * n) * (2 * cutoff)
    h *= np.kaiser(taps, 8.6)
    return (h / h.sum()).astype(np.float32)


def resample_to_16k(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono float32 PCM to the 16 kHz MuseTalk expects.

    48 kHz -> 16 kHz (the production path) is exactly 3:1 and uses the polyphase
    decimator. Other rates fall back to linear interpolation, which is ample for
    a debug preview and never silently aliases a production stream.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or sample_rate == SR16:
        return x
    if sample_rate % SR16 == 0:
        ratio = sample_rate // SR16
        h = _decimators.get(ratio)
        if h is None:
            h = _decimators[ratio] = _design_decimator(ratio)
        # mode="full" then dropping the filter's group delay keeps output sample
        # k aligned with input sample k*ratio, so consecutive blocks concatenate
        # onto one continuous 16 kHz timeline. Trimming to a whole number of
        # strides before decimating keeps the output length exactly N/ratio --
        # otherwise every block would gain ~2 samples and the 25 fps grid would
        # drift against the ingest timeline.
        y = np.convolve(x, h, mode="full")[len(h) // 2 :]
        y = y[: (x.size // ratio) * ratio]
        return np.ascontiguousarray(y[::ratio], dtype=np.float32)
    n_out = int(round(x.size * SR16 / sample_rate))
    return np.interp(
        np.arange(n_out, dtype=np.float64) * (sample_rate / SR16),
        np.arange(x.size, dtype=np.float64),
        x,
    ).astype(np.float32)


def _load_latents(path: Path):
    """Seam for the torch-free tests.

    Reading a `.pt` needs torch, which only exists in the GPU venv, but the
    rest of the avatar contract (which sidecar is which, coordinate
    propagation, the cache) is exactly what a GPU-less test can assert.
    Mirrors `musetalk_pipeline._save_latents`.
    """
    import torch

    # weights_only=True is the default and is what we want: this file is a
    # plain VAE latent tensor written by our own pipeline.
    return torch.load(path, map_location="cpu")


@dataclass
class _AvatarState:
    """One avatar's prepared artifacts, resolved from pack.prep."""

    bbox: list[int]
    base_bgr: np.ndarray  # HxWx3 uint8, canvas space -- blending works in BGR
    latents: object  # torch.Tensor [1,8,32,32] on device, weight dtype
    mask: np.ndarray  # HxW uint8, sized to mask_crop_box
    mask_crop_box: list[int]


@dataclass
class _SessionContext:
    """Per-session streaming state.

    100 ms at 25 fps is 2.5 frames, so rounding each block independently would
    silently emit 20 fps. Carrying the fractional remainder forward gives
    2,3,2,3,... -- 25 fps on average. RenderSession discards the engine's pts
    and only counts frames, so the accumulator is the lever that matters.
    """

    buf16k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    emitted: int = 0
    # Counted in whole input samples, not seconds: ten 100 ms blocks summed as
    # floats land on 0.9999999999999999, whose floor(*25) is 24 frames instead
    # of 25. Integer sample counts keep the frame grid exact.
    total_samples: int = 0
    # Pinned on first use so an LRU eviction cannot make a live session reload
    # its latents every block.
    avatar: _AvatarState | None = None


class _Runtime:
    """The global MuseTalk 1.5 weights, built once per engine (~3 GB at fp16)."""

    def __init__(self, models_dir: Path):
        import sys

        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))

        import torch
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE(model_path=str(models_dir / "sd-vae"))
        self.unet = UNet(
            unet_config=str(models_dir / "musetalkV15" / "musetalk.json"),
            model_path=str(models_dir / "musetalkV15" / "unet.pth"),
            device=self.device,
        )
        self.pe = PositionalEncoding(d_model=384)
        self.whisper = WhisperModel.from_pretrained(str(models_dir / "whisper"))
        self.whisper = self.whisper.to(device=self.device).eval()
        self.whisper.requires_grad_(False)
        self.audio_processor = AudioProcessor(feature_extractor_path=str(models_dir / "whisper"))
        self.feature_extractor = self.audio_processor.feature_extractor
        self.timesteps = torch.tensor([0], device=self.device)

        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.whisper = self.whisper.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype


class MuseTalkEngine(DriveEngine):
    name = "musetalk"

    # Relative to models_dir; checked by describe() and prepare() without
    # touching a tensor, so a misconfigured box fails loudly and early.
    WEIGHTS = (
        "musetalkV15/unet.pth",
        "musetalkV15/musetalk.json",
        "sd-vae/config.json",
        "whisper/config.json",
    )

    def __init__(self, models_dir: Path | None = None, fps: int | None = None):
        settings = get_settings()
        self.models_dir = (
            Path(models_dir) if models_dir else Path(settings.data_dir) / "models" / "musetalk"
        )
        self.fps = int(fps or settings.fps)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="musetalk")
        self._lock = threading.Lock()
        self._runtime: _Runtime | None = None
        self._avatars: OrderedDict[str, _AvatarState] = OrderedDict()
        self._closed = False

    # -- status ---------------------------------------------------------
    def _missing_weights(self) -> list[str]:
        return [w for w in self.WEIGHTS if not (self.models_dir / w).exists()]

    def describe(self) -> dict:
        """Cheap status for the admin System page. Never loads a weight."""
        return {
            "name": self.name,
            "gpu": gpu_available(),
            "models_dir": str(self.models_dir),
            "weights_present": not self._missing_weights(),
            "loaded": self._runtime is not None,
            "avatars_cached": len(self._avatars),
        }

    # -- lifecycle ------------------------------------------------------
    async def warmup(self) -> None:
        missing = self._missing_weights()
        if missing:
            raise EngineUnavailable(
                f"musetalk weights not found in {self.models_dir} (missing: {missing}); "
                "run scripts/fetch_musetalk_weights.sh"
            )
        await self._submit(self._warmup_sync)

    def _warmup_sync(self) -> None:
        import torch

        rt = self._ensure_runtime()
        latents = torch.zeros(1, 8, 32, 32, device=rt.device, dtype=rt.weight_dtype)
        prompt = torch.zeros(1, 50, 384, device=rt.device, dtype=rt.weight_dtype)
        with torch.no_grad():
            rt.unet.model(latents, rt.timesteps, encoder_hidden_states=rt.pe(prompt))
        if rt.device.type == "cuda":
            torch.cuda.synchronize()

    async def prepare(self, pack: AvatarPack) -> None:
        await self._submit(self._prepare_sync, pack)

    def _prepare_sync(self, pack: AvatarPack) -> None:
        self._ensure_avatar(self._ensure_runtime(), pack)

    def new_context(self, pack: AvatarPack) -> _SessionContext:
        return _SessionContext()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._avatars.clear()
        self._runtime = None
        await asyncio.get_running_loop().run_in_executor(None, self._pool.shutdown, True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- torch may be absent; nothing to free
            pass

    # -- driving --------------------------------------------------------
    async def drive(
        self,
        pack: AvatarPack,
        audio: np.ndarray,
        sample_rate: int,
        *,
        context: object | None = None,
    ) -> list[DriveFrame]:
        if self._closed:
            raise EngineUnavailable("musetalk engine is closed")
        ctx = context if isinstance(context, _SessionContext) else _SessionContext()
        return await self._submit(
            self._drive_sync, pack, np.asarray(audio, dtype=np.float32), sample_rate, ctx
        )

    def _drive_sync(
        self, pack: AvatarPack, audio: np.ndarray, sample_rate: int, ctx: _SessionContext
    ) -> list[DriveFrame]:
        rt = self._ensure_runtime()
        if ctx.avatar is None:
            ctx.avatar = self._ensure_avatar(rt, pack)
        avatar = ctx.avatar

        block16k = resample_to_16k(audio, sample_rate)
        if block16k.size:
            ctx.buf16k = np.concatenate((ctx.buf16k, block16k))[-MAX_BUF:]
        ctx.total_samples += int(audio.size)

        target = ctx.total_samples * self.fps // sample_rate
        first = ctx.emitted
        if target <= first:
            return []
        prompts = self._prompts_for(rt, ctx, first, target)
        ctx.emitted = target

        frames = []
        for i, image in enumerate(self._render(rt, avatar, prompts)):
            frames.append(DriveFrame(pts=(first + i) / self.fps, image=image))
        return frames

    def _prompts_for(self, rt: _Runtime, ctx: _SessionContext, first: int, target: int):
        """Whisper prompts for output frames [first, target)."""
        samples_per_frame = SR16 // self.fps  # 640 at 25 fps
        # Anchor the window on an output-frame boundary so window-relative frame
        # r maps exactly to absolute frame f0 + r.
        start = max(0, first - CONTEXT_FRAMES) * samples_per_frame
        buf = ctx.buf16k
        if start >= buf.size:
            start = max(0, buf.size - (SR16 // WHISPER_FPS))
        f0 = start // samples_per_frame
        window = buf[start:]
        if window.size < SR16 // WHISPER_FPS:
            # Never hand whisper an empty mel window; the frame indices below
            # clamp onto whatever the model returns.
            window = np.zeros(SR16 // WHISPER_FPS, dtype=np.float32)

        feats = rt.feature_extractor(
            window, return_tensors="pt", sampling_rate=SR16
        ).input_features
        prompts = rt.audio_processor.get_whisper_chunk(
            [feats], rt.device, rt.weight_dtype, rt.whisper, int(window.size), fps=self.fps
        )
        last = prompts.shape[0] - 1
        index = [min(max(f - f0, 0), last) for f in range(first, target)]
        return prompts[index]

    def _render(self, rt: _Runtime, avatar: _AvatarState, prompts) -> list[np.ndarray]:
        import cv2
        import torch
        from musetalk.utils.blending import get_image_blending

        x1, y1, x2, y2 = avatar.bbox
        size = (x2 - x1, y2 - y1)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, prompts.shape[0], BATCH_SIZE):
                wb = prompts[i : i + BATCH_SIZE].to(rt.device, dtype=rt.weight_dtype)
                latents = avatar.latents.repeat(wb.shape[0], 1, 1, 1)
                pred = rt.unet.model(
                    latents, rt.timesteps, encoder_hidden_states=rt.pe(wb)
                ).sample
                recon = rt.vae.decode_latents(pred.to(rt.vae.vae.dtype))
                for res_frame in recon:
                    res = cv2.resize(res_frame.astype(np.uint8), size)
                    # get_image_blending works in BGR and returns BGR; the media
                    # contract with the sink is RGB.
                    blended = get_image_blending(
                        avatar.base_bgr, res, avatar.bbox, avatar.mask, avatar.mask_crop_box
                    )
                    out.append(np.ascontiguousarray(blended[:, :, ::-1]))
        return out

    # -- model / avatar caches -----------------------------------------
    def _ensure_runtime(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is None:
                missing = self._missing_weights()
                if missing:
                    raise EngineUnavailable(
                        f"musetalk weights not found in {self.models_dir} (missing: {missing}); "
                        "run scripts/fetch_musetalk_weights.sh"
                    )
                self._runtime = _Runtime(self.models_dir)
        return self._runtime

    def _ensure_avatar(self, rt: _Runtime, pack: AvatarPack) -> _AvatarState:
        state = self._avatars.get(pack.avatar_id)
        if state is not None:
            self._avatars.move_to_end(pack.avatar_id)
            return state
        state = self._load_avatar(rt, pack)
        self._avatars[pack.avatar_id] = state
        while len(self._avatars) > AVATAR_CACHE_SIZE:
            self._avatars.popitem(last=False)
        return state

    def _load_avatar(self, rt: _Runtime, pack: AvatarPack) -> _AvatarState:
        from PIL import Image

        prep = pack.prep or {}
        if not prep.get("latents") or not prep.get("mask"):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} has no musetalk pack; prepare it with "
                "DH_AVATAR_PIPELINE=musetalk"
            )
        root = Path(pack.root)
        base = np.asarray(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        return _AvatarState(
            bbox=[int(v) for v in prep["bbox"]],
            base_bgr=np.ascontiguousarray(base[:, :, ::-1]),
            latents=latents,
            mask=np.asarray(Image.open(root / prep["mask"]).convert("L")),
            mask_crop_box=[int(v) for v in prep["mask_crop_box"]],
        )

    # -- plumbing -------------------------------------------------------
    async def _submit(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, functools.partial(fn, *args))
