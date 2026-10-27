"""The Wav2Lip audio path: the streaming mel and the window it hands the model.

What is pinned here is the *contract* with the network: the window's shape,
where it starts, and what fills the part of it that has not been spoken yet.
The mel itself is upstream's, and its numbers are asserted in the runtime
venv (the real mel needs librosa, which the torch-free venv does not have).
"""
from __future__ import annotations

import numpy as np
import pytest

from dh_server.wav2lip import audio as wav2lip_audio

SAMPLE_RATE = 48000
BLOCK = 4800  # 100 ms


def _counting_mel():
    """A stand-in mel: one frame per 200 samples, each bin counting.

    The value of a mel frame is its own index, so a test can tell which
    frames the window picked up.
    """

    def compute(wav: np.ndarray) -> np.ndarray:
        count = len(wav) // wav2lip_audio.HOP_SIZE
        if count <= 0:
            return np.zeros((0, wav2lip_audio.NUM_MELS), dtype=np.float32)
        base = np.arange(count, dtype=np.float32).reshape(-1, 1)
        return np.repeat(base, wav2lip_audio.NUM_MELS, axis=1)

    return compute


def test_the_window_is_sixteen_frames_of_eighty_bins():
    stream = wav2lip_audio.MelStream(fps=25, compute=_counting_mel())
    stream.push(np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE)
    window = stream.window(0)
    assert window.shape == (wav2lip_audio.MEL_WINDOW, wav2lip_audio.NUM_MELS)


def test_the_window_starts_at_the_frames_own_audio():
    """Upstream pairs frame i with the mel chunk that starts at i's own
    audio: ``mel[int(80 * i / fps)]``."""
    stream = wav2lip_audio.MelStream(fps=25, compute=_counting_mel())
    # 200 ms of audio: 3200 samples at 16 kHz, 15 mel frames.
    stream.push(np.zeros(SAMPLE_RATE // 5, dtype=np.float32), SAMPLE_RATE)

    assert stream.window(0)[0, 0] == 0.0
    assert stream.window(1)[0, 0] == 3.0  # int(80 * 1 / 25)
    assert stream.window(2)[0, 0] == 6.0
    assert stream.window(4)[0, 0] == 12.0  # int(80 * 4 / 25)


def test_the_unspoken_part_of_the_window_is_silence():
    """The window reaches 200 ms into the future; in a live stream that
    audio has not been spoken yet, so it is silence -- the same value
    upstream writes at the end of an audio file."""
    stream = wav2lip_audio.MelStream(fps=25, compute=_counting_mel())
    # 40 ms of audio: 640 samples at 16 kHz. A mel frame covers 800 samples
    # with 400 of them behind its start, so only the first two frames are
    # wholly inside the audio that has arrived.
    stream.push(np.zeros(1920, dtype=np.float32), SAMPLE_RATE)
    window = stream.window(0)
    assert window[0, 0] == 0.0  # the frame's own audio
    assert window[1, 0] == 1.0
    assert np.all(window[2:] == wav2lip_audio.SILENCE)


def test_the_silence_value_is_the_mel_of_silence():
    """-4 in every bin: upstream's normalisation clips there, and a silent
    waveform lands below the clip."""
    assert wav2lip_audio.SILENCE == -wav2lip_audio.MAX_ABS_VALUE
    assert wav2lip_audio.silence_mel().shape == (wav2lip_audio.NUM_MELS,)
    assert np.all(wav2lip_audio.silence_mel() == wav2lip_audio.SILENCE)


def test_the_mel_frame_count_follows_the_audio():
    stream = wav2lip_audio.MelStream(fps=25, compute=_counting_mel())
    stream.push(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    # One second of audio is 80 mel frames of 12.5 ms; the last two are not
    # derivable yet, because their window reaches past the audio.
    assert stream.mel_derived == 79


def test_the_window_of_a_frame_whose_audio_has_not_arrived_is_silence():
    stream = wav2lip_audio.MelStream(fps=25, compute=_counting_mel())
    stream.push(np.zeros(BLOCK, dtype=np.float32), SAMPLE_RATE)
    # Frame 100 is ten seconds away: none of its audio has been seen.
    window = stream.window(100)
    assert np.all(window == wav2lip_audio.SILENCE)


def test_the_resampler_brings_48k_down_to_16k():
    x = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = wav2lip_audio.resample_to_16k(x, SAMPLE_RATE)
    assert out.size == SAMPLE_RATE // 3
    # A whole number of mel hops, so the grid never drifts.
    assert out.size % wav2lip_audio.HOP_SIZE == 0


def test_the_resampler_leaves_16k_alone():
    x = np.ones(1000, dtype=np.float32)
    assert wav2lip_audio.resample_to_16k(x, 16000).size == 1000


def test_the_streaming_mel_is_the_batch_mel():
    """The frames the stream derives are the ones the whole-signal mel
    gives: a window whose audio has all arrived is identical either way."""
    pytest.importorskip("librosa")
    pytest.importorskip("scipy")
    wav = (np.sin(np.arange(16000) * 0.05) * 0.3).astype(np.float32)
    batch = wav2lip_audio.mel_spectrogram(wav)

    stream = wav2lip_audio.MelStream(fps=25)
    for start in range(0, len(wav), 1600):  # 100 ms blocks, like the session
        stream.push(wav[start : start + 1600], 16000)

    # Frame 15's window starts at mel frame 48 and ends at 63; the audio for
    # all of it has arrived, so the window is the batch mel's window.
    frame_index = 15
    start = int(wav2lip_audio.MEL_FPS * frame_index / 25)
    window = stream.window(frame_index)
    assert window.shape == batch[start : start + 16].shape
    assert np.allclose(window, batch[start : start + 16], atol=1e-4)


def test_the_real_mel_is_in_range():
    """The real mel needs librosa, so it only runs in the runtime venv."""
    pytest.importorskip("librosa")
    pytest.importorskip("scipy")
    wav = np.sin(np.arange(16000) * 0.05).astype(np.float32)
    mel = wav2lip_audio.mel_spectrogram(wav)
    assert mel.shape[1] == wav2lip_audio.NUM_MELS
    assert mel.min() >= -wav2lip_audio.MAX_ABS_VALUE
    assert mel.max() <= wav2lip_audio.MAX_ABS_VALUE
    # A loud sine has energy, so the mel is not the silence value.
    assert mel.max() > wav2lip_audio.SILENCE
