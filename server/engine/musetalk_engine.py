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
import concurrent.futures
import queue
import threading
import time
from collections import OrderedDict
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
# Frames per UNet call. Per-frame cost falls steeply to ~8 (15.6 -> 4.5 ms) and
# flattens after, so this is also the natural chunk for a multi-session batch.
BATCH_SIZE = 8
# Jobs coalesced into one batch. The worker drains whatever is already queued
# when it finishes a batch, so under load this is what lets two sessions share
# UNet calls instead of queueing behind each other. Eight 100 ms jobs is 20
# frames, which BATCH_SIZE then splits into three UNet calls.
MAX_JOBS_PER_BATCH = 8
# How long the worker waits for a second session's job before rendering alone.
#
# Two sessions only share a UNet call if both jobs are in the queue at once, and
# they are not: each session awaits its drive() before pushing the next block,
# so at most one job per session is ever in flight and the worker races the
# second one in. Without this wait a batch averages 1.05 jobs with two sessions
# -- i.e. batching silently never engages, which is exactly what the first run
# of the Stage 3 gate measured.
#
# It is only paid when a second session is actually active (see _concurrent),
# so a lone session's block latency is untouched.
GATHER_S = 0.004
# A context counts as active for this long after its last block.
CONTEXT_TTL_S = 5.0

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


def blend_face(
    base: np.ndarray,
    face: np.ndarray,
    bbox: list[int],
    mask: np.ndarray,
    crop_box: list[int],
) -> np.ndarray:
    """Composite `face` into `base` through `mask`, in whatever order they are in.

    Drop-in replacement for MuseTalk's `get_image_blending`, which does the same
    lerp through three full 720x1280 PIL round-trips per frame and costs 5.8 ms
    of a 100 ms block. This is a per-pixel lerp; numpy does it in 0.3 ms.

    MuseTalk's version flips BGR->RGB on the way into PIL and RGB->BGR on the
    way out. The flip is a per-pixel channel permutation and the mask is scalar
    per pixel, so the whole operation commutes with it: the result is correct in
    either order. `base` and `face` must agree with each other, and the return
    value comes back in that same order -- which is what lets the caller keep
    the whole pipeline in the sink's RGB and never flip a finished frame.

    Semantics, matching PIL's `paste(im, box, mask)`:
        out = region * (1 - m) + face_pasted * m
    """
    x, y, x1, y1 = (int(v) for v in bbox)
    x_s, y_s, x_e, y_e = (int(v) for v in crop_box)

    # Accumulate in place: the crop is blended once per frame, and every
    # temporary here is another pass over the region.
    out = base.copy()
    region = out[y_s:y_e, x_s:x_e].astype(np.float32)
    over = region.copy()
    over[y - y_s : y1 - y_s, x - x_s : x1 - x_s] = face

    m = mask.astype(np.float32)[..., None]
    m *= np.float32(1.0 / 255.0)
    region *= 1.0 - m
    region += over * m

    out[y_s:y_e, x_s:x_e] = np.clip(region + 0.5, 0.0, 255.0).astype(np.uint8)
    return out


def _decode_gpu(rt, pred):
    """MuseTalk's `VAE.decode_latents`, with the 8-bit quantisation on the GPU.

    The vendored version pulls the fp16 latents to the host and then does
    `/2+0.5`, `*255`, `round` and the uint8 cast in numpy -- four passes over an
    fp32 array per frame. On the device they are free, and the copy shrinks 4x
    because uint8 is what crosses the bus.

    Returns (b,3,H,W) uint8, still on the device, in RGB -- the order the sink
    contract wants, so nothing downstream ever flips a finished frame.
    """
    import torch

    vae = rt.vae
    latents = pred.to(vae.vae.dtype) / vae.scaling_factor
    image = vae.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return (image * 255).round().to(torch.uint8)


def _decode_frames(rt, pred):
    """`_decode_gpu` as HWC numpy -- the host-side view the probes compare against."""
    return _decode_gpu(rt, pred).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _blend_face_gpu(base, face, bbox, mask, crop_box):
    """`blend_face`, with the canvas resident on the device.

    Same lerp and same semantics; only the venue differs. In numpy every frame
    paid a 2.7 MB canvas copy, a crop-sized fp32 round trip and a clip/cast on
    the worker thread -- all of it serialised behind the GPU calls that had just
    produced the face, on the one thread that also feeds the UNet. Here the
    intermediates stay on the device and only the finished frame crosses back.

    `base` is (3,H,W) uint8, `face` (3,h,w) uint8, `mask` (1,ch,cw) float in
    [0,1] -- all on one device, all RGB.
    """
    import torch

    x, y, x1, y1 = (int(v) for v in bbox)
    x_s, y_s, x_e, y_e = (int(v) for v in crop_box)

    out = base.clone()
    out[:, y:y1, x:x1] = face
    over = out[:, y_s:y_e, x_s:x_e].to(torch.float32)
    under = base[:, y_s:y_e, x_s:x_e].to(torch.float32)
    under *= 1.0 - mask
    under += over * mask
    out[:, y_s:y_e, x_s:x_e] = (under + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
    return out


def _cat_prompts(parts: list):
    """Join per-job prompt tensors into one UNet batch.

    Seam for the torch-free tests, like `_load_latents`: production parts are
    torch tensors and only torch can concatenate them, but the batching logic
    that decides *which* parts go together is testable without a GPU.
    """
    import torch

    return torch.cat(parts, dim=0)


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
    base_rgb: np.ndarray  # HxWx3 uint8, canvas space, in the sink's RGB order
    latents: object  # torch.Tensor [1,8,32,32] on device, weight dtype
    mask: np.ndarray  # HxW uint8, sized to mask_crop_box
    mask_crop_box: list[int]
    # Device-side twins of base_rgb/mask, built on first render. Deferred out of
    # `_load_avatar` because the torch-free tests build this dataclass by hand
    # and must not need a device to do it.
    base_t: object = None
    mask_t: object = None


@dataclass
class _SessionContext:
    """Per-session streaming state.

    100 ms at 25 fps is 2.5 frames, so rounding each block independently would
    silently emit 20 fps. Carrying the fractional remainder forward gives
    2,3,2,3,... -- 25 fps on average. RenderSession discards the engine's pts
    and only counts frames, so the accumulator is the lever that matters.
    """

    buf16k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Absolute 16 kHz sample index of buf16k[0]. The buffer is a sliding window,
    # so its position in the session's timeline has to be carried alongside it:
    # frame indices are absolute, and computing a window offset without knowing
    # where the buffer starts is how a long session reads off the end of it.
    buf_start: int = 0
    # Total 16 kHz samples ever ingested; buf_start is derived from this.
    total16k: int = 0
    emitted: int = 0
    # Counted in whole input samples, not seconds: ten 100 ms blocks summed as
    # floats land on 0.9999999999999999, whose floor(*25) is 24 frames instead
    # of 25. Integer sample counts keep the frame grid exact.
    total_samples: int = 0
    # Pinned on first use so an LRU eviction cannot make a live session reload
    # its latents every block.
    avatar: _AvatarState | None = None


@dataclass
class _Job:
    """One session's drive() call, waiting for the batch worker."""

    pack: AvatarPack
    ctx: _SessionContext
    audio: np.ndarray
    sample_rate: int
    future: concurrent.futures.Future


@dataclass
class _Call:
    """A plain callable (warmup, prepare) run on the worker's thread."""

    fn: object
    args: tuple
    future: concurrent.futures.Future


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
        # One worker thread owns every CUDA call: it keeps the GPU serialized
        # and, because it drains whatever is queued when it finishes a batch,
        # it coalesces concurrent sessions into shared UNet calls.
        self._jobs: queue.Queue = queue.Queue()
        self._enqueue_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._worker_loop, name="musetalk-batch", daemon=True
        )
        self._worker.start()
        self._lock = threading.Lock()
        self._runtime: _Runtime | None = None
        self._avatars: OrderedDict[str, _AvatarState] = OrderedDict()
        # Session contexts seen recently, by id -> last block time. Touched only
        # on the worker thread (in _advance and _worker_loop), so it needs no
        # lock; it exists so a lone session never pays the gather window.
        self._ctx_seen: OrderedDict[int, float] = OrderedDict()
        self._closed = False

    # -- status ---------------------------------------------------------
    def _missing_weights(self) -> list[str]:
        return [w for w in self.WEIGHTS if not (self.models_dir / w).exists()]

    def pending(self) -> int:
        return self._jobs.qsize()

    def describe(self) -> dict:
        """Cheap status for the admin System page. Never loads a weight."""
        return {
            "name": self.name,
            "gpu": gpu_available(),
            "models_dir": str(self.models_dir),
            "weights_present": not self._missing_weights(),
            "loaded": self._runtime is not None,
            "avatars_cached": len(self._avatars),
            "pending": self._jobs.qsize(),
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
        # Flip the flag under the enqueue lock so no drive() can slip a job in
        # behind the sentinel, where it would wait forever on a dead worker.
        with self._enqueue_lock:
            self._closed = True
            self._jobs.put(None)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._worker.join)
        self._avatars.clear()
        self._runtime = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- torch may be absent; nothing to free
            pass

    def _enqueue(self, item) -> None:
        with self._enqueue_lock:
            if self._closed:
                raise EngineUnavailable("musetalk engine is closed")
            self._jobs.put(item)

    def _worker_loop(self) -> None:
        """The single thread that touches CUDA."""
        while True:
            item = self._jobs.get()
            if item is None:
                return
            if isinstance(item, _Call):
                self._resolve(item, lambda: item.fn(*item.args))
                continue
            batch = [item]
            # Drain what is already waiting, and -- when a second session is
            # live -- wait a moment for its job to land too, so the two share a
            # UNet call instead of queueing behind each other. With one session
            # this is a plain non-blocking drain.
            active = self._active()
            deadline = time.monotonic() + GATHER_S if active > 1 else None
            while len(batch) < MAX_JOBS_PER_BATCH:
                # A session cannot have two jobs in flight -- it awaits drive()
                # before pushing the next block -- so once the batch holds one
                # job per live session there is nothing left to wait for. Without
                # this the worker paid the full window on every batch, waiting
                # for a third job that could not exist.
                if deadline is None or len(batch) >= active:
                    try:
                        nxt = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self._jobs.get(timeout=remaining)
                    except queue.Empty:
                        break
                if nxt is None:
                    self._run_batch(batch)
                    return
                batch.append(nxt)
            self._run_batch(batch)

    def _active(self) -> int:
        """How many sessions have driven a block recently.

        Nothing tells the engine a session ended, so activity is aged out by
        timestamp. Touched only on the worker thread, so no lock is needed.
        """
        cutoff = time.monotonic() - CONTEXT_TTL_S
        while self._ctx_seen:
            _, last = next(iter(self._ctx_seen.items()))
            if last >= cutoff:
                break
            self._ctx_seen.popitem(last=False)
        return len(self._ctx_seen)

    def _concurrent(self) -> bool:
        """True while more than one session is live -- see `_active`."""
        return self._active() > 1

    def _run_batch(self, batch: list[_Job]) -> None:
        try:
            results = self._drive_sync(batch)
        except Exception as exc:  # noqa: BLE001 -- surfaced to every caller
            for job in batch:
                job.future.set_exception(exc)
            return
        for job, frames in zip(batch, results):
            job.future.set_result(frames)

    @staticmethod
    def _resolve(call: _Call, fn) -> None:
        try:
            call.future.set_result(fn())
        except Exception as exc:  # noqa: BLE001
            call.future.set_exception(exc)

    # -- driving --------------------------------------------------------
    async def drive(
        self,
        pack: AvatarPack,
        audio: np.ndarray,
        sample_rate: int,
        *,
        context: object | None = None,
    ) -> list[DriveFrame]:
        ctx = context if isinstance(context, _SessionContext) else _SessionContext()
        job = _Job(
            pack=pack,
            ctx=ctx,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            future=concurrent.futures.Future(),
        )
        self._enqueue(job)
        return await asyncio.wrap_future(job.future)

    def _advance(
        self, rt: _Runtime, job: _Job
    ) -> tuple[_AvatarState, int, object] | None:
        """Fold one block into its session's context and produce its prompts.

        Returns (avatar, first_frame, prompts), or None when the block was too
        short to complete an output frame.
        """
        ctx = job.ctx
        self._ctx_seen[id(ctx)] = time.monotonic()
        self._ctx_seen.move_to_end(id(ctx))
        if ctx.avatar is None:
            ctx.avatar = self._ensure_avatar(rt, job.pack)
        block16k = resample_to_16k(job.audio, job.sample_rate)
        if block16k.size:
            ctx.buf16k = np.concatenate((ctx.buf16k, block16k))[-MAX_BUF:]
            ctx.total16k += int(block16k.size)
            # Where the surviving window begins in the session's timeline.
            ctx.buf_start = ctx.total16k - int(ctx.buf16k.size)
        ctx.total_samples += int(job.audio.size)

        target = ctx.total_samples * self.fps // job.sample_rate
        first = ctx.emitted
        if target <= first:
            return None
        prompts = self._prompts_for(rt, ctx, first, target)
        ctx.emitted = target
        return ctx.avatar, first, prompts

    def _drive_sync(self, jobs: list[_Job]) -> list[list[DriveFrame]]:
        """Run one batch of jobs. Returns one frame list per job, in order.

        The per-session work (resample, window, whisper) is inherently serial,
        but the UNet is latency-bound -- 15.6 ms/frame at batch 3 against 4.5 at
        batch 8 -- so every job's frames are concatenated into as few UNet calls
        as the batch allows. That is the whole reason this takes a list.
        """
        rt = self._ensure_runtime()
        out: list[list[DriveFrame] | None] = [None] * len(jobs)
        work: list[tuple[int, _AvatarState, int, object]] = []
        for i, job in enumerate(jobs):
            advanced = self._advance(rt, job)
            if advanced is None:
                out[i] = []
                continue
            work.append((i, *advanced))

        if work:
            # Group by avatar: a batch shares one UNet call, so it must share
            # one set of latents. Two sessions on the same avatar do; two
            # different avatars cannot, and get a call each.
            groups: dict[int, list[int]] = {}
            for k, (_, avatar, _, _) in enumerate(work):
                groups.setdefault(id(avatar), []).append(k)
            for idxs in groups.values():
                avatar = work[idxs[0]][1]
                prompts = _cat_prompts([work[k][3] for k in idxs])
                images = self._render(rt, avatar, prompts)
                pos = 0
                for k in idxs:
                    i, _, first, pr = work[k]
                    n = int(pr.shape[0])
                    out[i] = [
                        DriveFrame(pts=(first + j) / self.fps, image=images[pos + j])
                        for j in range(n)
                    ]
                    pos += n
        return [frames if frames is not None else [] for frames in out]

    def _prompts_for(self, rt: _Runtime, ctx: _SessionContext, first: int, target: int):
        """Whisper prompts for output frames [first, target)."""
        samples_per_frame = SR16 // self.fps  # 640 at 25 fps
        # Anchor the window on an output-frame boundary so window-relative frame
        # r maps exactly to absolute frame f0 + r.
        start = max(0, first - CONTEXT_FRAMES) * samples_per_frame
        # The buffer is a sliding 2 s window, so a lookback that predates it must
        # be clamped to where the buffer actually starts -- not to `buf.size`,
        # which is a *length*. Getting this wrong handed whisper a 320-sample
        # window, and 320 samples at 16 kHz is 0.5 output frames, which makes
        # get_whisper_chunk return an empty list and torch.cat([]) throw. Any
        # session past ~2 s hit it.
        start = max(start, ctx.buf_start)
        f0 = start // samples_per_frame
        window = ctx.buf16k[start - ctx.buf_start :]
        if window.size < samples_per_frame:
            # Zero-pad to a whole output frame so get_whisper_chunk yields at
            # least one prompt. Padding goes at the end, which leaves frame f0
            # aligned with window offset 0 -- the mapping the indices below use.
            window = np.concatenate(
                (window, np.zeros(samples_per_frame - window.size, dtype=np.float32))
            )

        feats = rt.feature_extractor(
            window, return_tensors="pt", sampling_rate=SR16
        ).input_features
        prompts = rt.audio_processor.get_whisper_chunk(
            [feats], rt.device, rt.weight_dtype, rt.whisper, int(window.size), fps=self.fps
        )
        last = prompts.shape[0] - 1
        index = [min(max(f - f0, 0), last) for f in range(first, target)]
        return prompts[index]

    def _device_assets(self, rt: _Runtime, avatar: _AvatarState):
        """The canvas and mask as device tensors, built once per avatar."""
        import torch

        if avatar.base_t is None:
            avatar.base_t = (
                torch.from_numpy(avatar.base_rgb).permute(2, 0, 1).contiguous().to(rt.device)
            )
            avatar.mask_t = (
                torch.from_numpy(avatar.mask)
                .to(torch.float32)
                .div_(255.0)
                .unsqueeze(0)
                .to(rt.device)
            )
        return avatar.base_t, avatar.mask_t

    def _render(self, rt: _Runtime, avatar: _AvatarState, prompts) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        x1, y1, x2, y2 = avatar.bbox
        size = (y2 - y1, x2 - x1)  # interpolate takes (H, W)
        base_t, mask_t = self._device_assets(rt, avatar)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, prompts.shape[0], BATCH_SIZE):
                wb = prompts[i : i + BATCH_SIZE].to(rt.device, dtype=rt.weight_dtype)
                latents = avatar.latents.repeat(wb.shape[0], 1, 1, 1)
                pred = rt.unet.model(
                    latents, rt.timesteps, encoder_hidden_states=rt.pe(wb)
                ).sample
                recon = _decode_gpu(rt, pred)
                resized = (
                    F.interpolate(
                        recon.to(torch.float32),
                        size=size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .round()
                    .clamp_(0.0, 255.0)
                    .to(torch.uint8)
                )
                for j in range(resized.shape[0]):
                    frame = _blend_face_gpu(
                        base_t, resized[j], avatar.bbox, mask_t, avatar.mask_crop_box
                    )
                    out.append(frame.permute(1, 2, 0).contiguous().cpu().numpy())
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
        # Kept in RGB, the order the sink contract wants: the only frame that
        # would ever need flipping is the finished 720x1280 canvas, and at
        # 2.7 MB that flip cost more than the blend it fed.
        #
        # `np.array`, not `np.asarray`: PIL hands back a read-only buffer, and
        # the device twin below is a zero-copy view of whatever we store here.
        base_rgb = np.array(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        return _AvatarState(
            bbox=[int(v) for v in prep["bbox"]],
            base_rgb=base_rgb,
            latents=latents,
            mask=np.array(Image.open(root / prep["mask"]).convert("L")),
            mask_crop_box=[int(v) for v in prep["mask_crop_box"]],
        )

    # -- plumbing -------------------------------------------------------
    async def _submit(self, fn, *args):
        """Run `fn` on the worker thread, behind whatever batch is in flight."""
        call = _Call(fn=fn, args=args, future=concurrent.futures.Future())
        self._enqueue(call)
        return await asyncio.wrap_future(call.future)
