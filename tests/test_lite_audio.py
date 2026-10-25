"""The `lite` engine's audio path: resampling, the frame grid, the window.

These run in the torch-free ``.venv``: the audio path is numpy only, and
the encoder is injected. What is pinned here is the part that is easy to get
subtly wrong and hard to notice -- the window's *causality*.

FeatherTalk's own window reaches nine frames (360 ms) into the future. A
picture made from that window shows a mouth for audio that has not been
spoken yet, which on a live stream means the mouth lags the sound. The
server's window carries the current frame's tokens in the future slots
instead, and the model is trained the same way, so the two agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from server.lite.audio import (
    AUDIO_TOKENS,
    CURRENT_TOKEN_START,
    FUTURE_TOKEN_START,
    UNIT_SAMPLES,
    FeatherAudio,
    resample_to_16k,
)


class _FrameMeanEncoder:
    """A stand-in encoder: every frame's features carry that frame's mean.

    That makes a frame identifiable by its value, which is what lets the
    test say *which* frame a token came from.
    """

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        count = max(0, (len(waveform) - 80) // UNIT_SAMPLES)
        out = np.zeros((count, 1024), dtype=np.float32)
        for index in range(count):
            chunk = waveform[index * UNIT_SAMPLES : (index + 1) * UNIT_SAMPLES]
            out[index, :] = float(chunk.mean()) if chunk.size else 0.0
        return out


def test_resampling_48k_to_16k_keeps_the_frame_grid():
    pcm = np.arange(48000, dtype=np.float32)  # one second at 48 kHz

    out = resample_to_16k(pcm, 48000)

    # 3:1 exactly, so the 20 ms units stay aligned and the frame grid never
    # drifts over a long session.
    assert out.size == 16000
    assert out.size % UNIT_SAMPLES == 0


def test_resampling_16k_is_a_no_op():
    pcm = np.arange(1600, dtype=np.float32)

    assert np.array_equal(resample_to_16k(pcm, 16000), pcm)


def test_the_frame_grid_follows_the_sample_count():
    """The number of frames the audio covers is the sample count's doing."""
    audio = FeatherAudio(_FrameMeanEncoder())

    for block in range(1, 11):
        audio.push(np.full(4800, float(block), dtype=np.float32))
        expected = (block * 4800 * 25) // 48000
        assert audio.total_frames == expected
        for index in range(expected):
            assert audio.frame_available(index), f"frame {index} should be ready"
        assert not audio.frame_available(expected)


def test_the_window_is_forty_tokens():
    audio = FeatherAudio(_FrameMeanEncoder())
    for _ in range(4):
        audio.push(np.zeros(4800, dtype=np.float32))

    tokens = audio.window(2)

    assert tokens.shape == (AUDIO_TOKENS, 1024)


def test_the_future_slots_carry_the_current_frame():
    """The causal fill: the frames after the current one repeat it."""
    audio = FeatherAudio(_FrameMeanEncoder(), overlap_units=25)
    for block in range(6):
        audio.push(np.full(4800, 1000.0 + block, dtype=np.float32))

    tokens = audio.window(5)
    current = tokens[CURRENT_TOKEN_START : CURRENT_TOKEN_START + 2]

    # Nine frames of "future", two tokens each, all equal to the current
    # frame's two tokens.
    assert np.array_equal(tokens[FUTURE_TOKEN_START:], np.tile(current, (9, 1)))
    # And they are not simply equal to the current frame by accident: the
    # past carries the earlier audio.
    assert not np.array_equal(tokens[:CURRENT_TOKEN_START], current)


def test_the_past_is_zero_before_the_session_starts():
    audio = FeatherAudio(_FrameMeanEncoder())
    for _ in range(2):
        audio.push(np.full(4800, 7.0, dtype=np.float32))

    tokens = audio.window(0)

    # Frame 0 has no past at all.
    assert np.all(tokens[:CURRENT_TOKEN_START] == 0.0)


def test_a_window_beyond_the_arrived_audio_is_refused():
    audio = FeatherAudio(_FrameMeanEncoder())
    audio.push(np.zeros(4800, dtype=np.float32))

    with pytest.raises(ValueError):
        audio.window(9)


def test_the_level_estimate_follows_the_audio_slowly():
    """A per-window normalisation would make the feature scale drift.

    The upstream extractor normalises every call by that call's own mean and
    standard deviation; streaming has to estimate the level once instead, or
    the mouth twitches from block to block.
    """
    audio = FeatherAudio(_FrameMeanEncoder())

    quiet = np.full(4800, 0.01, dtype=np.float32)
    loud = np.full(4800, 1.0, dtype=np.float32)
    audio.push(quiet)
    after_quiet = audio._level_std
    for _ in range(20):
        audio.push(loud)
    after_loud = audio._level_std

    assert after_loud > after_quiet
    # Twenty blocks later it has moved, but not all the way: the estimate is
    # slow enough that one window never sees two different levels.
    assert after_loud < 1.0
