"""Streaming audio features for the `lite` model.

Two halves:

* **FeatherHuBERT** turns 16 kHz speech into 1024-dim features, one every
  20 ms. The encoder is a valid-convolution stack, so its receptive field
  reaches 400 samples past the start of each feature frame.
* **The audio window** assembles the 40 tokens the visual model expects for
  one output frame: 20 video frames (800 ms) centred on the current one.

Upstream's window is *not* causal -- it reaches 9 video frames (360 ms) into
the future, which would make the picture show a mouth for audio that has not
been spoken yet. This server drives a live stream whose lip-sync is anchored
on the audio clock the browser is playing, so the future slots carry the
current frame's tokens instead, and the model is **trained** the same way
(see :mod:`server.lite.dataset`). Training and inference therefore see the
same distribution.

Nothing here imports torch at module level: the encoder is passed in.
"""
from __future__ import annotations

import numpy as np

#: Samples per 20 ms at 16 kHz. One FeatherHuBERT feature per unit.
UNIT_SAMPLES = 320
#: The feature dimension FeatherHuBERT emits.
AUDIO_DIM = 1024
#: 20 ms units per video frame (40 ms at 25 fps).
UNITS_PER_FRAME = 2
#: Tokens per video frame: one per 20 ms unit.
TOKENS_PER_FRAME = UNITS_PER_FRAME
#: Video frames of context on each side of the current one.
AUDIO_HALF_WINDOW = 10
#: Tokens the visual model consumes: 20 frames x 2 tokens.
AUDIO_TOKENS = 2 * AUDIO_HALF_WINDOW * TOKENS_PER_FRAME
#: Where the current frame's tokens sit in the window. The window spans the
#: ten frames before and the nine after, so the current one is the eleventh
#: and its tokens are at 20 and 21.
CURRENT_TOKEN_START = AUDIO_HALF_WINDOW * TOKENS_PER_FRAME
#: The tokens that describe the future.
FUTURE_TOKEN_START = CURRENT_TOKEN_START + TOKENS_PER_FRAME

#: FeatherHuBERT's receptive field reaches 400 samples past a frame's start,
#: while frames are 320 samples apart -- so the newest frame needs 80 samples
#: that have not been spoken yet. They are held (the last 80 samples repeated)
#: rather than zeroed: a hold keeps the waveform continuous, a zero-fill puts
#: a step discontinuity through the convolution.
FEATURE_TAIL_SAMPLES = 80


def _design_decimator(ratio: int, taps: int = 33) -> np.ndarray:
    """A windowed-sinc low-pass for integer decimation by ``ratio``.

    Same design as the MuseTalk engine's decimator (Kaiser window, beta 8.6),
    reimplemented here so the `lite` model does not depend on the MuseTalk
    module -- a CPU-only serving box should not have to import MuseTalk's
    pipeline to decode audio.
    """
    cutoff = 1.0 / ratio
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n)
    h *= np.kaiser(taps, 8.6)
    return h / h.sum()


_decimators: dict[int, np.ndarray] = {}


def resample_to_16k(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Decimate ``x`` to 16 kHz.

    48 kHz -> 16 kHz is an exact 3:1 decimation and is what the session
    delivers. The output length is a whole number of 320-sample units so the
    frame grid never drifts.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if sample_rate == 16000:
        return x
    if sample_rate % 16000 == 0:
        ratio = sample_rate // 16000
        h = _decimators.get(ratio)
        if h is None:
            h = _design_decimator(ratio)
            _decimators[ratio] = h
        y = np.convolve(x, h, mode="full")[len(h) // 2 :]
        y = y[: (x.size // ratio) * ratio]
        return np.ascontiguousarray(y[::ratio], dtype=np.float32)
    # Not a whole decimation: fall back to linear interpolation. The session
    # always delivers 48 kHz, so this branch exists for tests and for a
    # deployment that changes DH_SAMPLE_RATE to something awkward.
    n_out = int(round(x.size * 16000 / sample_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, x.size - 1.0, num=x.size, dtype=np.float64)
    dst = np.linspace(0.0, x.size - 1.0, num=n_out, dtype=np.float64)
    return np.interp(dst, src, x).astype(np.float32)


class FeatherAudio:
    """Rolling FeatherHuBERT features for one session.

    The encoder is not causal in the signal-processing sense: a feature frame
    depends on 400 ms of context behind it, and the dilated TCN behind that
    reaches further still. Streaming therefore re-encodes a short window per
    block -- ``overlap_units`` of already-seen audio in front of the new
    samples -- and keeps only the frames that are new.

    ``overlap_units`` is the accuracy/cost dial: 25 units (500 ms) matches
    the window FeatherTalk's own C++ runtime uses, and costs one encoder pass
    over ~1 s of audio per 100 ms block.

    ``encoder`` is a callable that takes a **normalised** waveform and
    returns the features: ``[T, 1024]`` float32. It must not normalise the
    waveform itself -- see :meth:`_normalise` for why.
    """

    #: How fast the level estimate follows the incoming audio. One block is
    #: 100 ms, so 0.05 is a ~2 s time constant: slow enough that the gain
    #: barely moves inside one window, fast enough to follow a change of
    #: recording.
    LEVEL_ALPHA = 0.05

    def __init__(self, encoder, *, overlap_units: int = 25, history_units: int = 100):
        if overlap_units < 0:
            raise ValueError("overlap_units must not be negative")
        self._encoder = encoder
        self._overlap_units = overlap_units
        # How much audio to keep: the overlap in front of the newest block,
        # plus the history the window builder needs.
        self._history_units = max(history_units, overlap_units + 64)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0  # absolute 16 kHz sample index of buffer[0]
        self._total = 0  # absolute 16 kHz samples seen
        self._features = np.zeros((0, AUDIO_DIM), dtype=np.float32)
        self._feature_start = 0  # absolute feature index of features[0]
        self._level_mean = 0.0
        self._level_std = 1.0
        self._level_ready = False

    # -- properties -----------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Absolute 16 kHz samples seen."""
        return self._total

    @property
    def total_frames(self) -> int:
        """Absolute video frames the audio covers (40 ms each)."""
        return self._total // (UNIT_SAMPLES * UNITS_PER_FRAME)

    # -- level -----------------------------------------------------------

    def _update_level(self, samples: np.ndarray) -> None:
        """Track the loudness of the stream, slowly.

        The upstream feature extractor normalises each call by that call's
        own mean and standard deviation. Offline that is one normalisation
        for the whole file; streaming it would re-normalise every window, so
        the feature scale would drift from block to block and the mouth would
        twitch. Instead the level is estimated once, slowly, and applied to
        every window -- the same normalisation the training saw, minus the
        drift.
        """
        if samples.size == 0:
            return
        mean = float(samples.mean())
        std = float(np.sqrt(samples.var() + 1e-7))
        if not self._level_ready:
            self._level_mean = mean
            self._level_std = max(std, 1e-3)
            self._level_ready = True
            return
        alpha = self.LEVEL_ALPHA
        self._level_mean += alpha * (mean - self._level_mean)
        self._level_std += alpha * (std - self._level_std)

    def _normalise(self, chunk: np.ndarray) -> np.ndarray:
        """Scale a window with the tracked level."""
        if not self._level_ready:
            return chunk
        return (chunk - self._level_mean) / max(self._level_std, 1e-3)

    # -- ingestion ------------------------------------------------------

    def push(self, pcm: np.ndarray, sample_rate: int = 48000) -> None:
        """Add one block of audio and encode whatever became encodable."""
        samples = resample_to_16k(pcm, sample_rate)
        if samples.size == 0:
            return
        self._update_level(samples)
        self._buffer = np.concatenate((self._buffer, samples))
        self._total += int(samples.size)
        self._encode_new_frames()
        self._trim()

    def _encode_new_frames(self) -> None:
        """Encode the feature frames that the newest samples completed."""
        # Frame f covers samples [f*320, f*320+400). The newest frame we can
        # build is the one whose 320-sample span is complete; its last 80
        # samples are held (see FEATURE_TAIL_SAMPLES).
        last_frame = self._total // UNIT_SAMPLES - 1
        if last_frame < self._feature_start + self._features.shape[0]:
            return
        first_new = self._feature_start + self._features.shape[0]
        if last_frame < first_new:
            return

        # The window starts `overlap_units` before the first new frame so the
        # new frames see the same amount of left context the model was
        # trained with.
        window_first = max(0, first_new - self._overlap_units)
        start_sample = window_first * UNIT_SAMPLES
        end_sample = last_frame * UNIT_SAMPLES + UNIT_SAMPLES
        # ``self._buffer`` is the local buffer: it already starts at
        # ``self._buffer_start``, so the absolute sample index becomes an
        # offset into it.
        offset = start_sample - self._buffer_start
        if offset < 0:
            # The left context has already been trimmed away; encode what is
            # left and accept the shorter context.
            offset = 0
            start_sample = self._buffer_start
            window_first = start_sample // UNIT_SAMPLES
        chunk = self._buffer[offset : end_sample - self._buffer_start]
        if chunk.size == 0:
            return
        # Hold the tail so the newest frame's receptive field is complete.
        if chunk.size < (last_frame - window_first + 1) * UNIT_SAMPLES + FEATURE_TAIL_SAMPLES:
            if chunk.size < UNIT_SAMPLES:
                return
            hold = min(FEATURE_TAIL_SAMPLES, chunk.size)
            chunk = np.concatenate((chunk, chunk[-hold:]))

        encoded = self._encode(chunk)
        if encoded is None or encoded.shape[0] == 0:
            return
        # `encoded` starts at `window_first`; keep only the new frames.
        keep_from = first_new - window_first
        new = encoded[keep_from:]
        if new.shape[0] == 0:
            return
        if self._features.shape[0] == 0:
            self._features = new
            self._feature_start = first_new
        else:
            self._features = np.concatenate((self._features, new), axis=0)

    def _encode(self, chunk: np.ndarray) -> np.ndarray | None:
        """Run the encoder over a waveform and return its features."""
        try:
            features = self._encoder(self._normalise(chunk))
        except Exception:  # noqa: BLE001 -- a short buffer is not an error
            return None
        if features is None:
            return None
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 3 and features.shape[-1] == AUDIO_DIM:
            # [B, T, D] -> [T, D]
            features = features[0]
        if features.ndim != 2 or features.shape[-1] != AUDIO_DIM:
            return None
        return features

    def _trim(self) -> None:
        """Drop the audio and features nobody will ask for again."""
        keep_samples = self._history_units * UNIT_SAMPLES
        cutoff = self._total - keep_samples
        if cutoff > self._buffer_start:
            drop = cutoff - self._buffer_start
            self._buffer = self._buffer[drop:]
            self._buffer_start = cutoff
        keep_features = self._history_units * UNITS_PER_FRAME
        have = self._features.shape[0]
        if have > keep_features:
            drop = have - keep_features
            self._features = self._features[drop:]
            self._feature_start += drop

    # -- reading --------------------------------------------------------

    def frame_available(self, frame_index: int) -> bool:
        """Whether the audio for output frame ``frame_index`` has arrived."""
        needed = (frame_index + 1) * UNITS_PER_FRAME
        return self._feature_start + self._features.shape[0] >= needed

    def window(self, frame_index: int) -> np.ndarray:
        """The ``[40, 1024]`` token window for output frame ``frame_index``.

        The window spans video frames ``[frame_index - 10, frame_index + 10)``.
        Frames before the start of the session are zero; frames after the
        current one -- the future -- carry the current frame's tokens, which
        is what makes the window causal.
        """
        if not self.frame_available(frame_index):
            raise ValueError(
                f"audio for frame {frame_index} has not arrived yet "
                f"(have {self._feature_start + self._features.shape[0]} units)"
            )
        tokens = np.zeros((AUDIO_TOKENS, AUDIO_DIM), dtype=np.float32)
        first_frame = frame_index - AUDIO_HALF_WINDOW
        for offset in range(2 * AUDIO_HALF_WINDOW):
            source = first_frame + offset
            if source < 0:
                continue  # before the session started: zeros
            if source > frame_index:
                # In the future: reuse the current frame's tokens.
                source = frame_index
            tokens[offset * 2 : offset * 2 + 2] = self._tokens_for_frame(source)
        return tokens

    def _tokens_for_frame(self, frame_index: int) -> np.ndarray:
        """The two tokens of one video frame."""
        unit = frame_index * UNITS_PER_FRAME
        index = unit - self._feature_start
        if index < 0 or index + 2 > self._features.shape[0]:
            return np.zeros((2, AUDIO_DIM), dtype=np.float32)
        return self._features[index : index + 2]
