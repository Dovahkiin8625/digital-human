"""The causal audio window: training and serving must agree on it.

The serving path (:mod:`server.lite.audio`) builds the window frame by frame:
every future frame copies the current frame's two tokens, interleaved. The
training path has to produce the *same* window, or the model is served
inputs it never saw.

The fill must interleave two tokens into eighteen slots: a flat
assignment cannot broadcast them, and ``repeat`` would tile the pair
instead of interleaving it.

Needs torch and the vendored FeatherTalk checkout, so it runs in the runtime
venv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "vendor" / "FeatherTalk" / "dataset_mouth_roi.py").exists():
    pytest.skip("FeatherTalk is not vendored; run scripts/setup_lite.sh")

from server.lite.audio import (  # noqa: E402
    AUDIO_TOKENS,
    CURRENT_TOKEN_START,
    FUTURE_TOKEN_START,
)
from server.lite import vendor  # noqa: E402

vendor.ensure_on_path()

from server.lite.causal_dataset import causal_fill  # noqa: E402


def _window() -> "torch.Tensor":
    """A window whose tokens are all distinguishable."""
    values = np.arange(AUDIO_TOKENS, dtype=np.float32)[:, None]
    return torch.from_numpy(np.repeat(values, 8, axis=1))


def test_the_future_slots_repeat_the_current_frame_interleaved():
    window = _window()

    filled = causal_fill(window)

    current = filled[CURRENT_TOKEN_START : CURRENT_TOKEN_START + 2]
    future = filled[FUTURE_TOKEN_START:]
    # Nine frames of two tokens each, all equal to the current frame's two
    # tokens -- in pairs, not one token repeated.
    expected = current.repeat((9, 1))
    assert torch.equal(future, expected)
    # The two tokens of the current frame are different from each other, so
    # an interleaved fill is distinguishable from a flat one.
    assert current[0][0] != current[1][0]


def test_the_past_is_untouched():
    window = _window()

    filled = causal_fill(window)

    assert torch.equal(filled[:CURRENT_TOKEN_START], window[:CURRENT_TOKEN_START])


def test_the_fill_matches_the_serving_window():
    """The two paths must produce the same window.

    The serving path is the reference: it is what the model sees in
    production. A causal window is a fixed point of the training fill --
    running the fill over it changes nothing -- which is exactly what
    "the two agree" means.
    """
    from server.lite.audio import FeatherAudio

    class _Encoder:
        def __call__(self, waveform):
            count = max(0, (len(waveform) - 80) // 320)
            out = np.zeros((count, 1024), dtype=np.float32)
            for index in range(count):
                out[index, 0] = float(index)
                out[index, 1:] = float(index) * 1000.0
            return out

    audio = FeatherAudio(_Encoder(), overlap_units=25)
    for block in range(4):
        audio.push(np.full(4800, float(block), dtype=np.float32))
    served = torch.from_numpy(audio.window(2))

    refilled = causal_fill(served)

    assert torch.equal(refilled, served)
    # And the served window really is causal: its future is the current
    # frame's tokens.
    current = served[CURRENT_TOKEN_START : CURRENT_TOKEN_START + 2]
    assert torch.equal(served[FUTURE_TOKEN_START:], current.repeat((9, 1)))
