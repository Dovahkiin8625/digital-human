"""The `lite` drive engine: FeatherTalk on a CPU.

Turns a prepared `lite` pack plus PCM into lip-synced 720x1280 frames using
two small networks:

    FeatherHuBERT   3.4 M params   speech  -> 1024-d features every 20 ms
    visual UNet     5.5 M params   face + features -> repainted face patch

About 19 MB of weights and no GPU anywhere. MuseTalk, for comparison, is
947 M parameters of VAE + UNet + Whisper and wants a discrete card.

How a frame is made:

1. The session's audio is decoded to 16 kHz and encoded into features by
   FeatherHuBERT, over a rolling window that keeps half a second of context
   behind the newest audio.
2. The output frame's 40-token audio window is assembled around it. The
   *future* half of that window carries the current frame's tokens -- the
   model is trained that way too, so the picture never lags the sound
   (see :mod:`server.lite.audio`).
3. The frame's face crop is cut, masked and fed to the visual UNet together
   with the tokens; the painted patch is pasted back over the face region.

Torch is imported lazily, so this module imports in the torch-free `.venv`.

Every bit of work runs on one worker thread: the engine is CPU-bound, and
letting two sessions run at once would only make both slower.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from server.config import get_settings
from server.engine.base import (
    AvatarPack,
    DriveEngine,
    DriveFrame,
    EngineUnavailable,
)
from server.lite import geometry, pack as lite_pack, vendor

_log = logging.getLogger(__name__)


def cycle_index(frame: int, n: int) -> int:
    """Ping-pong index: 0, 1, ... n-1, n-2, ... 1, 0, 1, ...

    The source frames carry the head and body motion; the model only repaints
    the face. Walking them forward and back keeps the loop seamless.
    """
    if n <= 1:
        return 0
    period = 2 * n - 2
    k = frame % period
    return k if k < n else period - k


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR -> RGB.

    Everything inside the engine is BGR, because FeatherTalk was trained on
    OpenCV's channel order. The WebRTC track wants RGB, so the swap happens
    once, at the boundary.
    """
    return np.ascontiguousarray(frame[:, :, ::-1])


@dataclass
class _Call:
    """One job for the worker thread."""

    fn: object
    args: tuple
    future: concurrent.futures.Future


@dataclass
class _SessionContext:
    """Per-session streaming state.

    ``phase`` and ``emitted`` are read by ``RenderSession`` through
    ``hasattr``/``getattr`` to line the idle ping-pong up with the speech
    timeline, so they are part of the contract with the session layer.
    """

    emitted: int = 0
    phase: int = 0
    total_samples: int = 0
    audio: object | None = None  # server.lite.audio.FeatherAudio


@dataclass
class _AvatarState:
    """One avatar's baked assets: frames, face boxes, and its own UNet.

    The frames are held as JPEG bytes and decoded on demand: 375 canvas
    frames decoded would be a gigabyte of RAM, the bytes are 40 MB, and the
    decode is overlapped with the model's forward pass by the prefetcher
    thread (see :class:`LiteEngine`).
    """

    frames: list[bytes] = field(default_factory=list)
    coords: list[dict] = field(default_factory=list)
    visual: object | None = None
    cache_size: int = 24
    _cache: "OrderedDict[int, np.ndarray]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decoding: set = field(default_factory=set)

    def frame(self, index: int) -> np.ndarray:
        """The decoded source frame at ``index`` (BGR, uint8).

        BGR, not RGB: FeatherTalk was trained on OpenCV's channel order, and
        the conversion to RGB happens once, at the very end, when the frame
        goes to the WebRTC track.
        """
        with self._lock:
            hit = self._cache.get(index)
            if hit is not None:
                self._cache.move_to_end(index)
                return hit
        return self._decode(index)

    def prefetch(self, index: int) -> None:
        """Decode a frame into the cache without blocking the caller.

        Called from the prefetcher thread. A frame that is already cached --
        or already being decoded -- is left alone.
        """
        with self._lock:
            if index in self._cache or index in self._decoding:
                return
            self._decoding.add(index)
        try:
            image = self._decode_bytes(index)
        finally:
            with self._lock:
                self._decoding.discard(index)
        with self._lock:
            self._cache[index] = image
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _decode(self, index: int) -> np.ndarray:
        with self._lock:
            self._decoding.add(index)
        try:
            image = self._decode_bytes(index)
        finally:
            with self._lock:
                self._decoding.discard(index)
        with self._lock:
            self._cache[index] = image
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return image

    def _decode_bytes(self, index: int) -> np.ndarray:
        import cv2

        buffer = np.frombuffer(self.frames[index], dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise EngineUnavailable(f"lite frame {index} could not be decoded")
        return image


class LiteEngine(DriveEngine):
    """FeatherTalk, driven on the CPU."""

    name = "lite"

    def __init__(
        self,
        fps: int | None = None,
        *,
        overlap_units: int = 25,
        decode_cache: int = 16,
        encoder=None,
    ):
        settings = get_settings()
        self.fps = fps or settings.fps
        self._overlap_units = overlap_units
        self._decode_cache = decode_cache
        self._encoder = encoder  # injected in tests; loaded lazily otherwise
        self._avatars: "OrderedDict[str, _AvatarState]" = OrderedDict()
        self._cache_size = settings.avatar_cache_size
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name="lite-cpu", daemon=True)
        self._worker.start()
        # A second thread decodes the source frames while the model runs.
        # Decoding a 720x1280 JPEG is ~8 ms and the model's forward pass is
        # ~17 ms, so without this the decode would be a third of the frame
        # budget; with it, the decode hides behind the model. The queue is
        # bounded: a full queue means the decoder is behind, and dropping a
        # prefetch request is harmless -- the renderer will decode that
        # frame itself.
        self._prefetch: queue.Queue = queue.Queue(maxsize=64)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, name="lite-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    # -- prefetching ----------------------------------------------------

    def _prefetch_loop(self) -> None:
        while True:
            item = self._prefetch.get()
            if item is None:
                return
            state, index = item
            try:
                state.prefetch(index)
            except Exception as exc:  # noqa: BLE001 -- a bad frame is the renderer's problem
                _log.debug("lite prefetch of frame %s failed: %s", index, exc)

    # -- lifecycle ------------------------------------------------------

    async def warmup(self) -> None:
        """Load the shared weights.

        Raises :class:`EngineUnavailable` when the weights or the vendored
        checkout are missing -- the admin Models tab turns that into a
        "run this script" hint.
        """
        await self._submit(self._load_encoder)

    async def load(self) -> None:
        """The admin Models tab's entry point; same as :meth:`warmup`."""
        await self.warmup()

    def unload(self) -> None:
        """Release the shared weights and the per-avatar cache."""
        with self._lock:
            self._encoder = None
            self._avatars.clear()

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._jobs.put(None)
        self._prefetch.put(None)
        await asyncio.get_running_loop().run_in_executor(None, self._worker.join)

    # -- worker ---------------------------------------------------------

    def _enqueue(self, fn, *args) -> concurrent.futures.Future:
        call = _Call(fn=fn, args=args, future=concurrent.futures.Future())
        self._jobs.put(call)
        return call.future

    async def _submit(self, fn, *args):
        future = self._enqueue(fn, *args)
        return await asyncio.wrap_future(future)

    def _worker_loop(self) -> None:
        while True:
            call = self._jobs.get()
            if call is None:
                return
            try:
                result = call.fn(*call.args)
            except BaseException as exc:  # noqa: BLE001 -- outlive any one job
                if not call.future.cancelled():
                    call.future.set_exception(exc)
                _log.exception("lite worker: %s failed", getattr(call.fn, "__name__", call.fn))
            else:
                if not call.future.cancelled():
                    call.future.set_result(result)

    # -- assets ---------------------------------------------------------

    def _load_encoder(self):
        """Load FeatherHuBERT (the shared, non-personalised half).

        The result is a callable, not the module: it takes a normalised
        waveform and returns ``[T, 1024]`` features. The normalisation itself
        belongs to the caller (see :class:`server.lite.audio.FeatherAudio`),
        because upstream normalises per call and a streaming caller must not
        re-normalise every window.
        """
        if self._encoder is not None:
            return self._encoder
        missing = vendor.missing_weights()
        if missing:
            raise EngineUnavailable(
                f"lite weights not found in {vendor.weights_dir()} (missing: {missing}); "
                "run scripts/fetch_lite_weights.sh"
            )
        vendor.ensure_on_path()
        import torch
        from feather_hubert import get_feather_hubert_from_16k_speech, load_feather_hubert

        model = load_feather_hubert(
            str(vendor.weight_path(vendor.FEATHER_HUBERT)), device=torch.device("cpu")
        )
        device = torch.device("cpu")

        def encode(waveform: np.ndarray) -> np.ndarray:
            features = get_feather_hubert_from_16k_speech(
                waveform, model, device=device, normalize=False
            )
            return features.cpu().numpy()

        self._encoder = encode
        return encode

    def _prepare_sync(self, pack: AvatarPack) -> None:
        """Load one avatar's frames, face boxes and personalised UNet."""
        with self._lock:
            state = self._avatars.get(pack.avatar_id)
            if state is not None:
                self._avatars.move_to_end(pack.avatar_id)
                return
        self._load_encoder()

        root = Path(pack.root)
        coords = lite_pack.read_coords(root)
        frames_dir = lite_pack.frames_dir(root)
        names = sorted(p for p in frames_dir.glob("*.jpg"))
        if not names:
            raise EngineUnavailable(f"lite pack {root} has no frames")
        if len(coords) != len(names):
            raise EngineUnavailable(
                f"lite pack {root} is inconsistent: {len(names)} frames but "
                f"{len(coords)} face boxes"
            )
        frames = [p.read_bytes() for p in names]

        from server.lite.visual import VisualModel

        visual = VisualModel.load(lite_pack.visual_path(root), device="cpu")
        state = _AvatarState(
            frames=frames, coords=coords, visual=visual, cache_size=self._decode_cache
        )
        with self._lock:
            self._avatars[pack.avatar_id] = state
            self._avatars.move_to_end(pack.avatar_id)
            while len(self._avatars) > max(1, self._cache_size):
                self._avatars.popitem(last=False)

    async def prepare(self, pack: AvatarPack) -> None:
        await self._submit(self._prepare_sync, pack)

    def new_context(self, pack: AvatarPack) -> _SessionContext:
        from server.lite.audio import FeatherAudio

        # The encoder is resolved on the first block, not here: this method
        # runs on the event loop, and loading a checkpoint there would stall
        # every other session for a second.
        def encode(waveform: np.ndarray) -> np.ndarray:
            return self._load_encoder()(waveform)

        return _SessionContext(
            audio=FeatherAudio(encode, overlap_units=self._overlap_units)
        )

    # -- driving --------------------------------------------------------

    async def drive(
        self,
        pack: AvatarPack,
        audio: np.ndarray,
        sample_rate: int,
        *,
        context: object | None = None,
    ) -> list[DriveFrame]:
        ctx = context if isinstance(context, _SessionContext) else self.new_context(pack)
        return await self._submit(self._drive_sync, pack, np.asarray(audio, dtype=np.float32), sample_rate, ctx)

    def _drive_sync(
        self, pack: AvatarPack, audio: np.ndarray, sample_rate: int, ctx: _SessionContext
    ) -> list[DriveFrame]:
        state = self._avatar_state(pack)
        ctx.total_samples += int(audio.size)
        ctx.audio.push(audio, sample_rate)

        # The frame grid is driven by the sample count, not by the clock, so
        # 100 ms blocks of 48 kHz audio turn into 2 or 3 frames of 25 fps and
        # the average stays at 25.
        target = ctx.total_samples * self.fps // sample_rate
        first = ctx.emitted
        if target <= first:
            return []
        ctx.emitted = target

        # Ask the prefetcher for the frames that come next while this
        # block's frames are being painted. The decoder is faster than the
        # model, so a couple of frames of lead keeps the pipe full.
        count = len(state.frames)
        for index in range(first, target):
            for ahead in (1, 2, 3):
                nxt = cycle_index(ctx.phase + index + ahead, count)
                try:
                    self._prefetch.put_nowait((state, nxt))
                except queue.Full:
                    pass

        crops: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        sizes: list[tuple[int, int]] = []
        sources: list[np.ndarray] = []
        inputs: list[np.ndarray] = []
        tokens: list[np.ndarray] = []
        for index in range(first, target):
            resource = cycle_index(ctx.phase + index, count)
            source = state.frame(resource)
            bbox = tuple(int(v) for v in state.coords[resource]["bbox"])
            crop = geometry.crop_face(source, bbox)
            inner = geometry.extract_inner(crop)
            sources.append(source)
            crops.append(crop)
            boxes.append(bbox)
            size = state.coords[resource].get("size")
            sizes.append(tuple(size) if size else (bbox[2] - bbox[0], bbox[3] - bbox[1]))
            inputs.append(geometry.build_model_input(inner))
            tokens.append(ctx.audio.window(index))

        predictions = state.visual.predict_batch(
            np.stack(inputs), np.stack(tokens)
        )

        frames: list[DriveFrame] = []
        for offset, index in enumerate(range(first, target)):
            frame = sources[offset].copy()
            geometry.paste_prediction(
                frame, predictions[offset], crops[offset], boxes[offset], sizes[offset]
            )
            frames.append(DriveFrame(pts=index / self.fps, image=_to_rgb(frame)))
        return frames

    def _avatar_state(self, pack: AvatarPack) -> _AvatarState:
        with self._lock:
            state = self._avatars.get(pack.avatar_id)
            if state is not None:
                self._avatars.move_to_end(pack.avatar_id)
                return state
        # The avatar was evicted (or never prepared): load it and try again.
        self._prepare_sync(pack)
        with self._lock:
            state = self._avatars.get(pack.avatar_id)
        if state is None:  # pragma: no cover -- _prepare_sync raises first
            raise EngineUnavailable(f"lite pack for avatar {pack.avatar_id} could not be loaded")
        return state

    # -- idle -----------------------------------------------------------

    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The source frame at ping-pong position ``pos``.

        The frames carry the head and body motion, so an idle avatar still
        moves. Returns ``None`` when the pack cannot be loaded -- the session
        then falls back to its resting canvas.
        """
        try:
            state = self._avatar_state(pack)
        except Exception as exc:  # noqa: BLE001 -- an idle frame is never fatal
            _log.debug("lite idle frame unavailable for %s: %s", pack.avatar_id, exc)
            return None
        index = cycle_index(int(pos), len(state.frames))
        return _to_rgb(state.frame(index))

    def idle_position(self, context: object | None) -> int:
        if isinstance(context, _SessionContext):
            return int(context.phase + context.emitted)
        return 0

    # -- admin ----------------------------------------------------------

    def pending(self) -> int:
        """Jobs queued but not yet rendered."""
        return self._jobs.qsize()

    def describe(self) -> dict:
        """The Models tab's row. Never loads a weight."""
        loaded = self._encoder is not None
        missing = vendor.missing_weights()
        weights_dir = vendor.weights_dir()
        estimated = 0
        for name in vendor.WEIGHT_FILES:
            path = vendor.weight_path(name)
            try:
                estimated += path.stat().st_size
            except OSError:
                continue
        return {
            "name": self.name,
            "gpu": False,
            "models_dir": str(weights_dir),
            "weights_present": not missing,
            "loaded": loaded,
            "avatars_cached": len(self._avatars),
            "pending": self._jobs.qsize(),
            "models": [
                {
                    "id": "lite_runtime",
                    "kind": "drive_runtime",
                    "label_zh": "轻量驱动运行时 (FeatherTalk)",
                    "type": "drive_runtime",
                    "components": ["feather_hubert", "visual_unet"],
                    "source": str(weights_dir),
                    "estimated_bytes": estimated,
                    "loaded": loaded,
                    "can_load": not loaded and not missing,
                    "can_unload": loaded,
                    "loaded_at": None,
                }
            ],
        }
