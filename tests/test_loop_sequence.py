"""Contract tests for the forward-loop sequence assembler.

Frame placement, blend composition, the short-clip fallback and the
forward-only index are all plain numpy -- the engine-side selection of a
sequence index from an absolute frame counter is exactly the kind of
arithmetic that must stay here in the torch-free suite.
"""
import numpy as np
import pytest

from server.avatar.loop import (
    CROSSFADE_K,
    SHORT_CLIP_WARNING,
    build_loop_sequence,
    forward_index,
)


CANVAS_HW = (4, 4)


def _solid(value: int, *, shape=CANVAS_HW) -> np.ndarray:
    return np.full(shape + (3,), value, dtype=np.uint8)


def test_loop_sequence_appends_crossfade_back_to_head():
    # 20 distinct frames, K=8 default: 20 + 8 blend frames.
    frames = [_solid(i) for i in range(20)]
    out, warnings = build_loop_sequence(frames)

    assert warnings == []
    assert len(out) == 28
    # The last eight blend (tail -> head) frames are HWC averages: their centre
    # pixel is halfway between the source tail value and 0 (head value).
    last = out[-1]
    assert last[2, 2, 0] == pytest.approx((19 + 0) / 2.0, abs=1.0)


def test_loop_sequence_runs_prepare_on_each_blend_frame():
    seen = []

    def prepare(frame):
        seen.append(int(frame[2, 2, 0]))
        return _solid(99), []

    frames = [_solid(i) for i in range(20)]
    out, _ = build_loop_sequence(frames, prepare=prepare)
    # 8 blend frames in addition to the 20 pass-through frames; each is the
    # half-blend of `frames[-k]` and the head, so its centre pixel is the
    # half of the tail value (rounded).
    assert len(seen) == CROSSFADE_K
    assert seen == [6, 6, 7, 7, 8, 8, 9, 9]


def test_loop_sequence_falls_back_to_hard_cut_when_blend_prep_fails():
    def prepare(_frame):
        raise RuntimeError("no face")

    frames = [_solid(i) for i in range(20)]
    out, warnings = build_loop_sequence(frames, prepare=prepare)

    assert warnings == []
    assert len(out) == 28
    # Every blend frame becomes a hard cut to the head value 0.
    for blend in out[-CROSSFADE_K:]:
        assert int(blend[2, 2, 0]) == 0


def test_loop_sequence_short_clip_warns_and_skips_crossfade():
    # Fewer frames than K would otherwise ask for a negative slice.
    frames = [_solid(i) for i in range(5)]
    out, warnings = build_loop_sequence(frames)

    assert len(out) == 5
    assert SHORT_CLIP_WARNING in warnings


def test_loop_sequence_empty_is_empty():
    out, warnings = build_loop_sequence([])
    assert out == []
    assert warnings == []


def test_forward_index_wraps_without_reversing():
    # N=20: the legacy ping-pong would have period 38 with reversals; the
    # forward-only loop has period 20 and never steps backwards.
    seq = list(range(20))
    seen = [seq[forward_index(i, 20)] for i in range(60)]
    assert seen == seq + seq + seq


def test_forward_index_handles_zero_length():
    # A buggy pack (empty cycle) must not raise.
    assert forward_index(7, 0) == 0