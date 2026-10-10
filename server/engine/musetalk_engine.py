"""Real MuseTalk 1.5 drive engine.

Turns a prepared avatar pack plus PCM into lip-synced 720x1280 frames using the
vendored MuseTalk 1.5 models (VAE, UNet, whisper-tiny). No OpenMMLab: the only
mm* user in MuseTalk is DWPose, and the preparation pipeline already replaced it
with mediapipe, so nothing here imports mmcv/mmdet/mmpose/mmengine.

Model state is layered:

  * global weights   -- VAE/UNet/whisper, loaded once, lock-protected, warmed
                        at startup (cold start is ~34 s of loading + kernel
                        autotune, which must not land on the first session)
  * per-avatar cache -- base frames + per-frame latents/bbox/mask, LRU by
                        avatar_id. A video avatar holds N frames; an image
                        avatar is the N=1 case.

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
import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.avatar.musetalk_pipeline import IMPORT_LOCK, RUNTIME_WEIGHTS, VENDOR_DIR
from server.config import get_settings
from server.engine.base import (
    AvatarPack,
    DriveEngine,
    DriveFrame,
    EngineUnavailable,
    gpu_available,
)

_log = logging.getLogger(__name__)


def _settle(future: concurrent.futures.Future, *, result=None, exc=None) -> None:
    """Hand a result to a caller that may already be gone.

    `loop.run_in_executor`'s future is cancelled when the awaiting coroutine is
    cancelled -- a client that disconnects mid-block, a shutdown that cancels
    the warmup task. Setting a result on a cancelled future raises
    InvalidStateError, and letting that escape would kill the worker loop (see
    `_worker_loop`). The caller being gone is not an error: there is nobody left
    to tell.
    """
    if future.cancelled() or future.done():
        return
    try:
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)
    except concurrent.futures.InvalidStateError:
        pass  # cancelled between the check above and here


def cycle_index(frame: int, n: int) -> int:
    """Ping-pong index into an n-frame cycle: 0..n-1, n-1..1, 0..

    MuseTalk's own `frame_list_cycle = frames + frames[::-1]`, without storing
    the frames twice. n<=1 collapses to 0 so an image avatar needs no special case.
    """
    if n <= 1:
        return 0
    period = 2 * n - 2
    k = frame % period
    return k if k < n else period - k


def select_frame_assets(avatar: "_AvatarState", idxs: list[int]) -> list[tuple]:
    """Per output frame: (base, latent, bbox, mask, mask_box), in `idxs` order.

    Torch-free on purpose -- this is the feature's core selection step, and
    keeping it a plain function keeps it assertable in the default suite,
    where torch is absent.
    """
    return [
        (
            avatar.frames[c],
            avatar.latents[c],
            avatar.bboxes[c],
            avatar.masks[c],
            avatar.mask_boxes[c],
        )
        for c in idxs
    ]


# whisper and MuseTalk both work at 16 kHz; ingest is 48 kHz.
SR16 = 16000
# MuseTalk's whisper feature stream is 50 fps and each output frame consumes a
# 10-wide window of it (2 left + 2 right of padding, +/-1).
WHISPER_FPS = 50
# get_whisper_chunk zero-pads each side with zeros_like(feature[:, :N]); a
# slice on a feature shorter than N silently shrinks -- it does NOT pad -- so
# its 10-wide per-frame clip fails the width assert and the vendored code
# calls exit(). The left-side N is ceil(WHISPER_FPS/fps) * this multiplier,
# so the engine hands it at least that many REAL feature frames: 4 at 25 fps.
WHISPER_PAD_LEFT = 2
# Left context, in 25 fps output frames, replayed through whisper each block so
# the prompt for the first new frame is not computed off a truncated window.
CONTEXT_FRAMES = 5
# 2 s of 16 kHz audio: far more than CONTEXT_FRAMES plus one 100 ms block, and
# re-trimmed every call so the buffer cannot grow without bound.
MAX_BUF = SR16 * 2
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
    """One avatar's prepared artifacts, resolved from pack.prep.

    A video avatar carries N base frames and their per-frame sidecars; an image
    avatar is the N=1 case. `frames` stays on the CPU -- a 15 s avatar is ~1 GB
    -- and only the frames a batch needs are moved to the device.
    """

    frames: np.ndarray  # [N,H,W,3] uint8, canvas space, sink's RGB order
    latents: object  # torch.Tensor [N,8,32,32] on device, weight dtype
    bboxes: list[list[int]]  # length N
    masks: list[np.ndarray]  # length N, each sized to its frame's mask_crop_box
    mask_boxes: list[list[int]]  # length N

    @property
    def cycle_len(self) -> int:
        return len(self.bboxes)


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
    # Ping-pong cycle position of the frame just before this context's first
    # speech frame. Speech frames read cycle_index(phase + emitted + j), so a
    # session whose idle track reached position p starts speaking on that same
    # frame. Whisper/audio bookkeeping is untouched: it is anchored on emitted.
    phase: int = 0
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

        # Shared with the avatar pipeline: whichever thread gets here first does
        # the first `diffusers` import, the other waits. See IMPORT_LOCK.
        with IMPORT_LOCK:
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

    # The runtime set `_Runtime` actually loads, relative to models_dir and
    # checked by describe()/warmup() without touching a tensor. Shared with the
    # avatar pipeline (see RUNTIME_WEIGHTS) so the two cannot disagree about
    # whether the weights are present.
    WEIGHTS = RUNTIME_WEIGHTS

    def __init__(self, models_dir: Path | None = None, fps: int | None = None):
        settings = get_settings()
        self.models_dir = (
            Path(models_dir) if models_dir else Path(settings.data_dir) / "models" / "musetalk"
        )
        self.fps = int(fps or settings.fps)
        self.cache_size = int(settings.avatar_cache_size)
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

    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The cached avatar's source frame at cycle position `pos`.

        start() already ran prepare() on the worker thread, so the avatar is
        cached; an LRU eviction is repaired by re-running prepare -- also on
        the worker. Never build or load here, on the event-loop thread, where
        it would stall every session. Anything going wrong returns None: a
        keep-alive tick must not break the session, and the caller falls back
        to its static canvas.
        """
        try:
            avatar = self._avatars.get(pack.avatar_id)
            if avatar is None:
                await self.prepare(pack)
                avatar = self._avatars.get(pack.avatar_id)
            if avatar is None:
                return None
            return avatar.frames[cycle_index(int(pos), avatar.cycle_len)]
        except Exception as exc:  # noqa: BLE001
            # Never break the keep-alive loop; leave a trace for diagnosis.
            _log.debug("idle frame unavailable for %s: %s", pack.avatar_id, exc, exc_info=True)
            return None

    def idle_position(self, context: object | None) -> int:
        if isinstance(context, _SessionContext):
            return int(context.phase + context.emitted)
        return 0

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
        """The single thread that touches CUDA.

        Nothing may escape this loop. It is the process's only CUDA thread, and
        there is no supervisor to restart it: an exception that unwinds out of
        here leaves every later `drive()` waiting on a future that will never be
        resolved, i.e. a permanently hung engine and no recovery but a restart.
        """
        while True:
            item = self._jobs.get()
            if item is None:
                return
            try:
                # The drain in _dispatch shares this queue, so it -- not the
                # `get()` above -- may be the one that pulls the sentinel.
                if self._dispatch(item):
                    return
            except BaseException:  # noqa: BLE001 -- outlive any one job
                # BaseException, not Exception: vendored MuseTalk calls exit()
                # (SystemExit) when its whisper clip width check fails. The
                # batch layer below settles the in-flight futures before one
                # can reach here, but that directory is not code we trust to
                # stay well-behaved -- an unwound worker is a hung engine.
                _log.exception("musetalk worker: unhandled error, continuing")

    def _dispatch(self, item) -> bool:
        """Run one queued item, coalescing concurrent jobs into one batch.

        Returns True when the stop sentinel was pulled while draining. `close()`
        enqueues `None` and then joins this thread; the drain loop below reads
        the same queue, so if it consumed the sentinel silently the worker would
        loop back to a blocking `get()` and never return -- `close()`'s join()
        would never complete, and lifespan shutdown would hang the process
        until it was killed. Reporting the sentinel here is what lets the worker
        exit instead of swallowing it.
        """
        if isinstance(item, _Call):
            self._resolve(item, lambda: item.fn(*item.args))
            return False
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
                return True
            batch.append(nxt)
        self._run_batch(batch)
        return False

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
        except BaseException as exc:  # noqa: BLE001 -- fail the callers
            # BaseException on purpose: a SystemExit out of the vendored audio
            # processor must fail these calls, not kill the worker -- and with
            # it every session queued behind them.
            for job in batch:
                _settle(job.future, exc=exc)
            return
        for job, frames in zip(batch, results):
            _settle(job.future, result=frames)

    @staticmethod
    def _resolve(call: _Call, fn) -> None:
        try:
            _settle(call.future, result=fn())
        except BaseException as exc:  # noqa: BLE001
            # Same reason as _run_batch: an exit() from vendor code during a
            # prepare/describe call must fail the call, not kill the worker.
            _settle(call.future, exc=exc)

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
                cycle_idxs: list[int] = []
                for k in idxs:
                    job_i, _, first, pr = work[k]
                    # Every job uses its OWN session's phase: two sessions share
                    # the avatar and the UNet call, never the cycle position.
                    ctx_phase = int(getattr(jobs[job_i].ctx, "phase", 0))
                    cycle_idxs += [
                        cycle_index(ctx_phase + first + j, avatar.cycle_len)
                        for j in range(int(pr.shape[0]))
                    ]
                prompts = _cat_prompts([work[k][3] for k in idxs])
                images = self._render(rt, avatar, cycle_idxs, prompts)
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
        # Real feature must be long enough that the vendor's zeros_like pad
        # slices do not shrink: ceil(WHISPER_FPS/fps) * WHISPER_PAD_LEFT real
        # 50 Hz frames = 4 * 320 = 1280 samples at 25 fps.
        min_feature_frames = (
            (WHISPER_FPS + self.fps - 1) // self.fps * WHISPER_PAD_LEFT
        )
        min_window_samples = min_feature_frames * (SR16 // WHISPER_FPS)
        # Anchor the window on an output-frame boundary so window-relative frame
        # r maps exactly to absolute frame f0 + r.
        start = max(0, first - CONTEXT_FRAMES) * samples_per_frame
        # The buffer is a sliding 2 s window, so a lookback that predates it must
        # be clamped to where the buffer actually starts -- not to `buf.size`,
        # which is a *length*. Getting this wrong handed whisper a 320-sample
        # window, and 320 samples at 16 kHz is 0.5 output frames, which makes
        # get_whisper_chunk return an empty list and torch.cat([]) throw. Any
        # session past ~2 s hit it.
        #
        # Clamp *up* to the next frame boundary at or after buf_start, never
        # backwards: audio before buf_start is gone. `max(start, ctx.buf_start)`
        # looks equivalent but is not -- buf_start is a multiple of the
        # 1600-sample ingest block and 1600 is not a multiple of the 640-sample
        # output frame (1600/640 = 2.5), so on every other block it left `start`
        # half a frame (20 ms) off the frame grid and f0, hence every prompt
        # label, alternated block to block. Dropping at most one frame from the
        # head is harmless: the window carries CONTEXT_FRAMES of left context
        # and the frames actually needed sit at its end.
        frame_start = ((ctx.buf_start + samples_per_frame - 1) // samples_per_frame) * samples_per_frame
        start = max(start, frame_start)
        f0 = start // samples_per_frame
        window = ctx.buf16k[start - ctx.buf_start :]
        if window.size < min_window_samples:
            # Zero-pad so the real feature is long enough for get_whisper_chunk's
            # side padding to be full width (see WHISPER_PAD_LEFT). One output
            # frame is NOT enough: a 640-sample window gives a 2-row feature,
            # every pad slice shrinks to those 2 rows, and the 10-wide clip
            # asserts and exit()s -- SystemExit, on the worker thread. Padding
            # goes at the end, leaving frame f0 at window offset 0 -- the
            # mapping the indices below use. The extra prompts describe
            # silence; the frame math above selects only indices the real
            # audio actually covers.
            window = np.concatenate(
                (window, np.zeros(min_window_samples - window.size, dtype=np.float32))
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

    def _to_device(self, rt: _Runtime, rgb: np.ndarray):
        import torch

        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(rt.device)

    def _mask_to_device(self, rt: _Runtime, mask: np.ndarray):
        import torch

        return (
            torch.from_numpy(mask).to(torch.float32).div_(255.0).unsqueeze(0).to(rt.device)
        )

    def _render(self, rt: _Runtime, avatar: _AvatarState, idxs: list[int], prompts) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, prompts.shape[0], BATCH_SIZE):
                wb = prompts[i : i + BATCH_SIZE].to(rt.device, dtype=rt.weight_dtype)
                chunk = select_frame_assets(
                    avatar, [idxs[i + j] for j in range(wb.shape[0])]
                )
                lb = torch.stack([a[1] for a in chunk])
                pred = rt.unet.model(lb, rt.timesteps, encoder_hidden_states=rt.pe(wb)).sample
                recon = _decode_gpu(rt, pred)
                for j in range(recon.shape[0]):
                    base, _, bbox, mask, mask_box = chunk[j]
                    x1, y1, x2, y2 = bbox
                    face = (
                        F.interpolate(
                            recon[j : j + 1].to(torch.float32),
                            size=(y2 - y1, x2 - x1),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .round()
                        .clamp_(0.0, 255.0)
                        .to(torch.uint8)[0]
                    )
                    frame = _blend_face_gpu(
                        self._to_device(rt, base),
                        face,
                        bbox,
                        self._mask_to_device(rt, mask),
                        mask_box,
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
        while len(self._avatars) > self.cache_size:
            self._avatars.popitem(last=False)
        return state

    def _load_avatar(self, rt: _Runtime, pack: AvatarPack) -> _AvatarState:
        prep = pack.prep or {}
        if not prep.get("latents") or not (prep.get("coords") or prep.get("mask")):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} has no musetalk pack; prepare it with "
                "DH_AVATAR_PIPELINE=musetalk"
            )
        root = Path(pack.root)
        mode = prep.get("mode")
        if mode == "video":
            return self._load_video_avatar(rt, pack, root, prep)
        if mode == "image":
            return self._load_image_avatar(rt, root, prep)
        return self._load_legacy_avatar(rt, root, prep)

    def _load_image_avatar(self, rt: _Runtime, root: Path, prep: dict) -> _AvatarState:
        """A photo prepared under the unified layout: N=1, no video to rebuild.

        Unlike a video pack the base frame cannot be re-decoded from the
        source (cv2.VideoCapture does not open PNG/WebP), so the canvas and
        mask written at prepare time are the load source.
        """
        from PIL import Image

        base_rgb = np.array(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        mask = np.array(
            Image.open(root / prep["masks"] / "00000000.png").convert("L")
        )
        return _AvatarState(
            frames=base_rgb[None, ...],
            latents=latents,
            bboxes=[[int(v) for v in prep["bbox"]]],
            masks=[mask],
            mask_boxes=[[int(v) for v in prep["mask_crop_box"]]],
        )

    def _load_legacy_avatar(self, rt: _Runtime, root: Path, prep: dict) -> _AvatarState:
        """A single-frame pack written before video support: N=1."""
        from PIL import Image

        base_rgb = np.array(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        return _AvatarState(
            frames=base_rgb[None, ...],
            latents=latents,
            bboxes=[[int(v) for v in prep["bbox"]]],
            masks=[np.array(Image.open(root / prep["mask"]).convert("L"))],
            mask_boxes=[[int(v) for v in prep["mask_crop_box"]]],
        )

    def _load_video_avatar(self, rt: _Runtime, pack: AvatarPack, root: Path, prep: dict) -> _AvatarState:
        import json

        from PIL import Image

        from server.avatar.video import decode_frames

        settings = get_settings()
        meta = json.loads((root / prep["dir"] / "meta.json").read_text())
        kept = [int(i) for i in meta["kept"]]
        coords = json.loads((root / prep["coords"]).read_text())
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        masks = [
            np.array(Image.open(root / prep["masks"] / f"{i:08d}.png").convert("L"))
            for i in range(len(kept))
        ]
        decoded, _ = decode_frames(
            root / prep["source"],
            target_fps=self.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if not (
            len(kept) == len(coords) == int(latents.shape[0]) == len(masks) == int(meta["frame_count"])
        ) or not kept or max(kept) >= len(decoded):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} video pack is inconsistent "
                f"(kept={len(kept)}, coords={len(coords)}, latents={int(latents.shape[0])}, "
                f"decoded={len(decoded)})"
            )
        return _AvatarState(
            frames=np.ascontiguousarray(decoded[kept]),
            latents=latents,
            bboxes=[[int(v) for v in c["bbox"]] for c in coords],
            masks=masks,
            mask_boxes=[[int(v) for v in c["mask_crop_box"]] for c in coords],
        )

    # -- plumbing -------------------------------------------------------
    async def _submit(self, fn, *args):
        """Run `fn` on the worker thread, behind whatever batch is in flight."""
        call = _Call(fn=fn, args=args, future=concurrent.futures.Future())
        self._enqueue(call)
        return await asyncio.wrap_future(call.future)
