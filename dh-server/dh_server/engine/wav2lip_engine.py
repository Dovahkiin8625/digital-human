"""The `wav2lip` drive engine: Wav2Lip on a CPU.

Turns a prepared `wav2lip` pack plus PCM into lip-synced 720x1280 frames:

    mel spectrogram   the session's audio, 16 frames per output frame
    generator         36.3 M parameters, one 96x96 face crop in, one out

The model is **generic**: one checkpoint drives every avatar, so the engine
holds one network and a per-avatar cache of frames and face boxes. About
146 MB of weights and no GPU anywhere. MuseTalk, for comparison, is 947 M
parameters of VAE + UNet + Whisper and wants a discrete card.

How a frame is made:

1. The session's audio is decoded to 16 kHz and turned into a mel
   spectrogram (see :mod:`dh_server.wav2lip.audio`). The window of a frame
   starts at the frame's own audio and reaches 200 ms forward; the part that
   has not been spoken yet is silence, so the mouth is always on the sound
   that is playing now.
2. The frame's face crop is cut out of the source frame and handed to the
   network together with the mel window.
3. The painted face is pasted back over the frame's face region.

While the session has nothing to say, the *idle* frames are served instead:
the same motion, repainted at build time with a silent mel window, so the
avatar keeps moving without moving its mouth (see :mod:`dh_server.wav2lip.idle`).

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

from dh_server.config import get_settings
from dh_server.engine.base import (
    AvatarPack,
    DriveEngine,
    DriveFrame,
    EngineUnavailable,
)
from dh_server.wav2lip import geometry, pack as wav2lip_pack, vendor
from dh_server.wav2lip.model import pick_device

_log = logging.getLogger(__name__)


def cycle_index(frame: int, n: int) -> int:
    """Ping-pong index: 0, 1, ... n-1, n-2, ... 1, 0, 1, ...

    The source frames carry the head and body motion; the model only
    repaints the face. Walking them forward and back keeps the loop seamless.
    """
    if n <= 1:
        return 0
    period = 2 * n - 2
    k = frame % period
    return k if k < n else period - k


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR -> RGB.

    Everything inside the engine is BGR, because Wav2Lip was trained on
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
    mel: object | None = None  # dh_server.wav2lip.audio.MelStream


@dataclass
class _FrameSet:
    """A set of JPEG-encoded frames, decoded on demand.

    The frames are held as JPEG bytes and decoded when they are needed: 375
    canvas frames decoded would be a gigabyte of RAM, the bytes are 40 MB,
    and the decode is overlapped with the model's forward pass by the
    prefetcher thread (see :class:`Wav2LipEngine`).
    """

    frames: list[bytes] = field(default_factory=list)
    cache_size: int = 24
    _cache: "OrderedDict[int, np.ndarray]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decoding: set = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, index: int) -> np.ndarray:
        """The decoded frame at ``index`` (BGR, uint8)."""
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
            raise EngineUnavailable(f"wav2lip frame {index} could not be decoded")
        return image


@dataclass
class _AvatarState:
    """One avatar's baked assets: frames, idle frames and face boxes."""

    frames: _FrameSet = field(default_factory=_FrameSet)
    idle: _FrameSet | None = None
    coords: list[dict] = field(default_factory=list)

    def frame(self, index: int) -> np.ndarray:
        """The decoded motion frame at ``index`` (BGR, uint8)."""
        return self.frames.frame(index)


class Wav2LipEngine(DriveEngine):
    """Wav2Lip, driven on the CPU."""

    name = "wav2lip"

    def __init__(self, fps: int | None = None, *, decode_cache: int = 16, model=None):
        settings = get_settings()
        self.fps = fps or settings.fps
        self._decode_cache = decode_cache
        # An injected model stands for the loaded one, so the device is only
        # needed when this engine has to load the weights itself.
        self._device = "cpu" if model is not None else pick_device()
        self._model = model  # injected in tests; loaded lazily otherwise
        self._avatars: "OrderedDict[str, _AvatarState]" = OrderedDict()
        self._cache_size = settings.avatar_cache_size
        self._jobs: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name="wav2lip-cpu", daemon=True)
        self._worker.start()
        # A second thread decodes the source frames while the model runs.
        # Decoding a 720x1280 JPEG is ~8 ms and the model's forward pass is
        # ~20 ms, so without this the decode would be a third of the frame
        # budget; with it, the decode hides behind the model. The queue is
        # bounded: a full queue means the decoder is behind, and dropping a
        # prefetch request is harmless -- the renderer will decode that
        # frame itself.
        self._prefetch: queue.Queue = queue.Queue(maxsize=64)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, name="wav2lip-prefetch", daemon=True
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
                state.frames.prefetch(index)
            except Exception as exc:  # noqa: BLE001 -- a bad frame is the renderer's problem
                _log.debug("wav2lip prefetch of frame %s failed: %s", index, exc)

    # -- lifecycle ------------------------------------------------------

    async def warmup(self) -> None:
        """Load the shared weights.

        Raises :class:`EngineUnavailable` when the weights or the vendored
        checkout are missing -- the admin Models tab turns that into a
        "run this script" hint.
        """
        await self._submit(self._load_model)

    async def load(self) -> None:
        """The admin Models tab's entry point; same as :meth:`warmup`."""
        await self.warmup()

    def unload(self) -> None:
        """Release the shared weights and the per-avatar cache."""
        with self._lock:
            self._model = None
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
                _log.exception("wav2lip worker: %s failed", getattr(call.fn, "__name__", call.fn))
            else:
                if not call.future.cancelled():
                    call.future.set_result(result)

    # -- assets ---------------------------------------------------------

    def _load_model(self):
        """Load the generator (the shared, non-personalised half).

        The result is a callable, not the module: it takes a batch of face
        crops and a batch of mel windows and returns the painted faces.
        """
        if self._model is not None:
            return self._model
        missing = vendor.missing_weights()
        if missing:
            raise EngineUnavailable(
                f"wav2lip weights not found in {vendor.weights_dir()} (missing: {missing}); "
                "run scripts/fetch_wav2lip_weights.sh"
            )
        from dh_server.wav2lip.model import Wav2LipModel

        self._model = Wav2LipModel(
            vendor.weight_path(vendor.GENERATOR), device=self._device
        )
        # The audio stack is imported here rather than on the first frame:
        # importing librosa takes seconds, and the first frame of the first
        # session is the worst possible place for that.
        from dh_server.wav2lip.audio import warm_up

        warm_up()
        return self._model

    def _prepare_sync(self, pack: AvatarPack) -> None:
        """Load one avatar's frames and face boxes."""
        with self._lock:
            state = self._avatars.get(pack.avatar_id)
            if state is not None:
                self._avatars.move_to_end(pack.avatar_id)
                return
        self._load_model()

        root = Path(pack.root)
        coords = wav2lip_pack.read_coords(root)
        frames_dir = wav2lip_pack.frames_dir(root)
        names = sorted(p for p in frames_dir.glob("*.jpg"))
        if not names:
            raise EngineUnavailable(f"wav2lip pack {root} has no frames")
        if len(coords) != len(names):
            raise EngineUnavailable(
                f"wav2lip pack {root} is inconsistent: {len(names)} frames but "
                f"{len(coords)} face boxes"
            )
        frames = [p.read_bytes() for p in names]
        state = _AvatarState(
            frames=_FrameSet(frames=frames, cache_size=self._decode_cache),
            idle=self._load_idle_frames(root, len(frames)),
            coords=coords,
        )
        with self._lock:
            self._avatars[pack.avatar_id] = state
            self._avatars.move_to_end(pack.avatar_id)
            while len(self._avatars) > max(1, self._cache_size):
                self._avatars.popitem(last=False)

    async def prepare(self, pack: AvatarPack) -> None:
        await self._submit(self._prepare_sync, pack)

    def _load_idle_frames(self, root: Path, expected: int) -> "_FrameSet | None":
        """The pack's closed-mouth track, when it has one.

        It is optional: a pack built before the idle track existed has no
        idle directory, and the session then shows its resting canvas. A
        track that does not line up with the motion frames is ignored too:
        the idle position and the speech position have to be the same
        frame, or the avatar would jump when it starts talking.
        """
        names = sorted(wav2lip_pack.idle_frames_dir(root).glob("*.jpg"))
        if not names:
            return None
        if len(names) != expected:
            _log.warning(
                "wav2lip pack %s has %d idle frames for %d motion frames; "
                "ignoring the idle track",
                root,
                len(names),
                expected,
            )
            return None
        return _FrameSet(
            frames=[p.read_bytes() for p in names], cache_size=self._decode_cache
        )

    def new_context(self, pack: AvatarPack) -> _SessionContext:
        from dh_server.wav2lip.audio import MelStream

        return _SessionContext(mel=MelStream(fps=self.fps))

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
        return await self._submit(
            self._drive_sync, pack, np.asarray(audio, dtype=np.float32), sample_rate, ctx
        )

    def _drive_sync(
        self, pack: AvatarPack, audio: np.ndarray, sample_rate: int, ctx: _SessionContext
    ) -> list[DriveFrame]:
        state = self._avatar_state(pack)
        ctx.total_samples += int(audio.size)
        ctx.mel.push(audio, sample_rate)

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

        sources: list[np.ndarray] = []
        crops: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        mels: list[np.ndarray] = []
        for index in range(first, target):
            resource = cycle_index(ctx.phase + index, count)
            source = state.frame(resource)
            box = tuple(int(v) for v in state.coords[resource]["bbox"])
            sources.append(source)
            crops.append(geometry.crop_face(source, box))
            boxes.append(box)
            mels.append(ctx.mel.window(index))

        predictions = self._load_model().predict(np.stack(crops), np.stack(mels))

        frames: list[DriveFrame] = []
        for offset, index in enumerate(range(first, target)):
            frame = sources[offset].copy()
            patch = (predictions[offset] * 255.0).round().clip(0, 255).astype(np.uint8)
            geometry.paste_prediction(frame, patch, boxes[offset])
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
            raise EngineUnavailable(f"wav2lip pack for avatar {pack.avatar_id} could not be loaded")
        return state

    # -- idle -----------------------------------------------------------

    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The idle frame at ping-pong position ``pos``.

        The idle track is the pack's motion frames repainted with a silent
        mel window: the head and body motion are the source video's, but the
        mouth is closed. The pack's own frames cannot be used here -- they
        come from a *speaking* video, and playing them back would show the
        avatar talking while the session is silent.

        Returns ``None`` when the pack has no idle track (it was built
        before the track existed) or cannot be loaded -- the session then
        falls back to its resting canvas.
        """
        try:
            state = self._avatar_state(pack)
        except Exception as exc:  # noqa: BLE001 -- an idle frame is never fatal
            _log.debug("wav2lip idle frame unavailable for %s: %s", pack.avatar_id, exc)
            return None
        if state.idle is None or not len(state.idle):
            return None
        index = cycle_index(int(pos), len(state.idle))
        return _to_rgb(state.idle.frame(index))

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
        loaded = self._model is not None
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
            "gpu": self._device != "cpu",
            "models_dir": str(weights_dir),
            "weights_present": not missing,
            "loaded": loaded,
            "avatars_cached": len(self._avatars),
            "pending": self._jobs.qsize(),
            "models": [
                {
                    "id": "wav2lip_runtime",
                    "kind": "drive_runtime",
                    "label_zh": "Wav2Lip 驱动运行时",
                    "type": "drive_runtime",
                    "components": ["wav2lip_generator", "s3fd_detector"],
                    "source": str(weights_dir),
                    "estimated_bytes": estimated,
                    "loaded": loaded,
                    "can_load": not loaded and not missing,
                    "can_unload": loaded,
                    "loaded_at": None,
                }
            ],
        }
