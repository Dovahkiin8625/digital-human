"""Streaming mel features for the Wav2Lip model.

Wav2Lip consumes a **mel spectrogram**, not the raw waveform: one 16-frame
window of 80 mel bins per output frame. The numbers below are the upstream
`hparams.py` verbatim -- they are not free parameters, the network was trained
against exactly them.

    sample rate   16000 Hz      the only rate the model sees
    n_fft          800          50 ms of audio per STFT frame
    hop size       200          12.5 ms between frames -> 80 mel frames/s
    mel bins        80          fmin 55 Hz, fmax 7600 Hz
    pre-emphasis  0.97         upstream's `lfilter([1, -0.97], [1], wav)`

The window a frame consumes starts at the frame's **own** audio and reaches
200 ms into the future (upstream: ``mel[int(80 * frame / fps) : +16]``). A
live stream does not have the future yet, so the frames that have not arrived
are filled with the mel of silence -- the same value upstream writes at the
end of an audio file, and the same value a real pause in the speech produces.
The mouth therefore stays on the sound that is playing now.

Nothing here imports torch, librosa or scipy at module level: the torch-free
`.venv` must still be able to import this module.
"""
from __future__ import annotations

import numpy as np

#: Everything upstream's `hparams.py` fixes, restated here.
SAMPLE_RATE = 16000
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
NUM_MELS = 80
FMIN = 55
FMAX = 7600
PREEMPHASIS = 0.97

#: The normalisation upstream applies to the decibel mel: a fixed affine map
#: onto [-4, 4], not a per-utterance mean/std.
MIN_LEVEL_DB = -100.0
REF_LEVEL_DB = 20.0
MAX_ABS_VALUE = 4.0

#: Mel frames per second of audio: 16000 / 200.
MEL_FPS = SAMPLE_RATE // HOP_SIZE

#: The mel frames one output frame consumes.
MEL_WINDOW = 16

#: The mel value of silence. Upstream's normalisation clips to +-4, and a
#: silent waveform lands at -120 dB, which maps below the clip -- so silence
#: is exactly -4 in every bin.
SILENCE = -MAX_ABS_VALUE

#: How much of the mel history to keep. The window of the frame being
#: rendered reaches 16 mel frames back from its own start, so half a second
#: of history is ample.
HISTORY_FRAMES = 40


def _design_decimator(ratio: int, taps: int = 33) -> np.ndarray:
    """A windowed-sinc low-pass for integer decimation by ``ratio``.

    Same design as the MuseTalk engine's decimator (Kaiser window, beta 8.6),
    reimplemented here so the Wav2Lip model does not depend on the MuseTalk
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
    delivers. The output length is a whole number of 200-sample units so the
    mel grid never drifts.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if sample_rate == SAMPLE_RATE:
        return x
    if sample_rate % SAMPLE_RATE == 0:
        ratio = sample_rate // SAMPLE_RATE
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
    n_out = int(round(x.size * SAMPLE_RATE / sample_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, x.size - 1.0, num=x.size, dtype=np.float64)
    dst = np.linspace(0.0, x.size - 1.0, num=n_out, dtype=np.float64)
    return np.interp(dst, src, x).astype(np.float32)


#: The mel filter bank, built once. Upstream caches it the same way.
_mel_basis = None


def _basis():
    """The mel filter bank, built on first use and then kept."""
    global _mel_basis
    if _mel_basis is None:
        import librosa

        _mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX
        )
    return _mel_basis


def warm_up() -> None:
    """Import the audio stack and build the mel filter bank.

    Importing librosa takes seconds, the filter bank is fixed and the first
    STFT builds its own machinery, so all three are done when the model is
    loaded rather than on the first frame a session asks for.
    """
    _basis()
    mel_spectrogram(np.zeros(N_FFT + HOP_SIZE, dtype=np.float32))


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """The mel spectrogram of a raw waveform, as ``[frames, NUM_MELS]``.

    Upstream's `audio.melspectrogram`, transposed: upstream keeps the mel in
    ``[bins, frames]`` and the model is fed that way; this server keeps the
    time axis first, because the frames of a stream arrive in order.

    The computation is upstream's, step for step: pre-emphasis, then the STFT,
    then the mel filter bank, then decibels, then the fixed normalisation.
    """
    from scipy import signal

    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return np.zeros((0, NUM_MELS), dtype=np.float32)

    # Upstream's `preemphasis(wav, 0.97)`: `lfilter([1, -k], [1], wav)`.
    wav = signal.lfilter([1, -PREEMPHASIS], [1], wav)
    return mel_from_preemphasized(wav)


def mel_from_preemphasized(wav: np.ndarray) -> np.ndarray:
    """The mel spectrogram of an already pre-emphasized waveform.

    Split out from :func:`mel_spectrogram` because the streaming path
    pre-emphasizes as the audio arrives -- the filter has to see the samples
    in order -- and then hands the pre-emphasized buffer here. Pre-emphasis
    applied twice would not be upstream's signal.
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return np.zeros((0, NUM_MELS), dtype=np.float32)

    # Upstream's `_stft`: librosa's STFT, complex, [bins, frames].
    stft = librosa.stft(y=wav, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE)

    # Upstream's `_linear_to_mel` + `_amp_to_db`.
    basis = _basis()
    min_level = 10.0 ** (MIN_LEVEL_DB / 20.0)
    decibels = 20.0 * np.log10(np.maximum(min_level, np.abs(basis @ np.abs(stft))))
    decibels = decibels - REF_LEVEL_DB

    # Upstream's `_normalize`: a fixed map onto [-4, 4].
    normalised = (2.0 * MAX_ABS_VALUE) * (
        (decibels - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)
    ) - MAX_ABS_VALUE
    normalised = np.clip(normalised, -MAX_ABS_VALUE, MAX_ABS_VALUE)
    return np.ascontiguousarray(normalised.T, dtype=np.float32)


def silence_mel() -> np.ndarray:
    """The mel of silence: ``-4`` in every bin."""
    return np.full(NUM_MELS, SILENCE, dtype=np.float32)


class MelStream:
    """The rolling mel spectrogram of one session.

    The waveform is kept in a sliding window and the mel is recomputed over
    that window whenever new audio arrives. Only the frames whose whole STFT
    window lies inside the *real* signal are kept: librosa zero-pads the
    edges of whatever it is given, which is what upstream's file start does
    too, but a trimmed buffer's left edge is not a file start, and a frame
    that borrowed from it would be wrong.
    """

    def __init__(self, fps: int = 25, *, compute=None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        # The mel function is injectable so the framing and the window can be
        # asserted without librosa (which is not in the torch-free venv).
        # It takes the *pre-emphasized* buffer, like
        # :func:`mel_from_preemphasized`.
        self._compute = compute or mel_from_preemphasized
        self._buf = np.zeros(0, dtype=np.float64)
        self._buf_start = 0  # absolute 16 kHz sample index of _buf[0]
        self._total = 0  # absolute 16 kHz samples seen
        self._mel = np.zeros((0, NUM_MELS), dtype=np.float32)
        self._mel_start = 0  # absolute mel index of _mel[0]
        self._mel_derived = 0  # mel frames derived since the session started
        self._pe_state = 0.0  # the pre-emphasis filter's last input sample

    # -- properties -----------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Absolute 16 kHz samples seen."""
        return self._total

    @property
    def mel_frames(self) -> int:
        """How many mel frames are retained."""
        return self._mel.shape[0]

    @property
    def mel_derived(self) -> int:
        """How many mel frames have been derived since the session started."""
        return self._mel_derived

    # -- ingestion ------------------------------------------------------

    def push(self, pcm: np.ndarray, sample_rate: int = 48000) -> None:
        """Add one block of audio and derive whatever mel became derivable."""
        samples = resample_to_16k(pcm, sample_rate)
        if samples.size == 0:
            return
        samples = self._preemphasize(samples)
        self._buf = np.concatenate((self._buf, samples))
        self._total += int(samples.size)
        self._refresh_mel()
        self._trim()

    def _preemphasize(self, x: np.ndarray) -> np.ndarray:
        """`lfilter([1, -0.97], [1], x)`, continued from the last block.

        The filter has no feedback: ``a = [1]``, so the output is
        ``x[n] - 0.97 * x[n-1]`` -- an FIR whose state is the previous
        *input* sample. Streaming it is therefore exact: the block continues
        from where the previous one stopped, which is what filtering the
        whole stream at once would have produced.

        It is written out rather than delegated to ``scipy.signal.lfilter``
        so the module stays importable (and testable) in the torch-free
        venv, which has no scipy. A 100 ms block is 1600 samples, so the
        loop costs a fraction of a millisecond.
        """
        k = PREEMPHASIS
        prev = self._pe_state
        out: list[float] = []
        for value in x.tolist():
            out.append(value - k * prev)
            prev = value
        self._pe_state = prev
        return np.asarray(out, dtype=np.float64)

    def _refresh_mel(self) -> None:
        """Recompute the mel over the buffer and keep the frames that are new."""
        if self._buf.size == 0:
            return
        mel = self._compute(self._buf)
        if mel.shape[0] == 0:
            return
        # The buffer starts on a mel boundary, so buffer frame k is the
        # absolute mel frame _buf_start // HOP_SIZE + k.
        base = self._buf_start // HOP_SIZE
        # A frame is exact when its whole 800-sample window is inside the
        # real signal. At the very start of the session the buffer's left edge
        # IS the signal's left edge, so the zero padding there is upstream's
        # own file-start padding and the frame is kept.
        first_k = 0 if self._buf_start == 0 else (N_FFT // 2 + HOP_SIZE - 1) // HOP_SIZE
        # The right edge moves with the audio: a frame whose window reaches
        # past what has arrived is not derivable yet.
        last_k = (self._buf.size - N_FFT // 2) // HOP_SIZE
        if last_k < first_k:
            return
        have = self._mel_start + self._mel.shape[0]
        wanted_from = max(base + first_k, have)
        wanted_to = base + last_k
        if wanted_to < wanted_from:
            return
        fresh = mel[wanted_from - base : wanted_to - base + 1]
        if fresh.shape[0] == 0:
            return
        if self._mel.shape[0] == 0:
            self._mel = fresh
            self._mel_start = wanted_from
        else:
            self._mel = np.concatenate((self._mel, fresh), axis=0)
        self._mel_derived += fresh.shape[0]

    def _trim(self) -> None:
        """Drop the audio and mel nobody will ask for again."""
        keep_samples = (HISTORY_FRAMES * HOP_SIZE + N_FFT) * 2
        cutoff = self._total - keep_samples
        # The cut lands on a mel boundary, so the buffer still starts one.
        cutoff -= cutoff % HOP_SIZE
        if cutoff > self._buf_start:
            drop = cutoff - self._buf_start
            self._buf = self._buf[drop:]
            self._buf_start = cutoff
        keep_frames = HISTORY_FRAMES
        if self._mel.shape[0] > keep_frames:
            drop = self._mel.shape[0] - keep_frames
            self._mel = self._mel[drop:]
            self._mel_start += drop

    # -- reading --------------------------------------------------------

    def window(self, frame_index: int) -> np.ndarray:
        """The ``[MEL_WINDOW, NUM_MELS]`` mel window of output frame ``frame_index``.

        The window starts at the frame's own audio -- upstream's
        ``mel[int(80 * frame / fps)]`` -- and runs 200 ms forward. The frames
        that have not arrived yet are filled with silence, which is what
        upstream writes at the end of an audio file and what a real pause in
        the speech produces.
        """
        start = int(MEL_FPS * frame_index / self.fps)
        out = np.empty((MEL_WINDOW, NUM_MELS), dtype=np.float32)
        have = self._mel_start + self._mel.shape[0]
        for offset in range(MEL_WINDOW):
            index = start + offset
            if index < self._mel_start:
                out[offset] = 0.0  # before the session started
            elif index < have:
                out[offset] = self._mel[index - self._mel_start]
            else:
                out[offset] = SILENCE
        return out
