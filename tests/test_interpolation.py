"""Frame-rate remap planning for sub-25 fps avatar sources.

The torch part of RIFE only exists in the GPU venv, but *which* source frames
each output frame blends, and at what timestep, is plain arithmetic -- and it
is the part a wrong sign or rounding rule would desync, so it stays here in
the torch-free suite.
"""
import numpy as np
import pytest

from server.avatar.interpolation import (
    SUB25_WARNING,
    interpolate_timeline,
    rife_weights_path,
    timeline_plan,
)


def test_24_to_25_uses_arbitrary_timesteps_between_source_frames():
    # 25 source frames at 24 fps span 25/24 s -> 26 output slots.
    plan = timeline_plan(25, 24.0, 25, max_frames=375)
    assert len(plan) == 26
    assert plan[0] == (0, 0.0)
    # Output k=1 lands at 0.96 source-frame periods: 96 % of the way 0 -> 1.
    assert plan[1] == (0, pytest.approx(0.96, abs=1e-9))
    assert plan[2] == (1, pytest.approx(0.92, abs=1e-9))
    # k=25 lands exactly on source frame 24 -> passthrough, not 24@t=1.
    assert plan[25] == (24, 0.0)
    # Every non-passthrough timestep stays strictly inside (0, 1).
    assert all(0.0 < t < 1.0 for _, t in plan if t)


def test_equal_framerate_is_identity_passthrough():
    plan = timeline_plan(40, 25.0, 25, max_frames=375)
    assert plan == [(k, 0.0) for k in range(40)]


def test_higher_source_fps_maps_onto_source_intervals_too():
    # The planner is generic; the pipeline only calls it for slow sources, but
    # the math must not assume src < dst.
    plan = timeline_plan(12, 30.0, 25, max_frames=375)
    assert len(plan) == 10
    assert plan[0] == (0, 0.0)
    assert plan[1] == (1, pytest.approx(0.2, abs=1e-9))
    assert plan[9] == (10, pytest.approx(0.8, abs=1e-9))


def test_plan_caps_output_length():
    plan = timeline_plan(1000, 24.0, 25, max_frames=375)
    assert len(plan) == 375


def test_plan_clamps_past_the_last_source_frame():
    # 24 source frames at 24 fps = 1.0 s -> 25 outputs. k=24 wants 23.04,
    # i.e. a frame 24 that does not exist: clamp to the last real frame.
    plan = timeline_plan(24, 24.0, 25, max_frames=375)
    assert len(plan) == 25
    assert plan[-1] == (23, 0.0)


def test_timestep_just_below_one_snaps_to_the_next_frame():
    # Float placement can put a timestamp 1e-16 short of an exact source hit;
    # that must read as the exact frame, not a t~=1 interpolation call.
    plan = timeline_plan(50, 24.0, 25, max_frames=375)
    assert plan[25] == (24, 0.0)


class _FakeRife:
    """Records (i, t) calls and returns a tagged frame."""

    def __init__(self):
        self.calls = []

    def interpolate(self, a, b, timestep):
        ia = int(a[0, 0, 0])
        ib = int(b[0, 0, 0])
        self.calls.append((ia, ib, round(float(timestep), 6)))
        out = np.full_like(a, 99)
        out[0, 0, 1] = int(round(timestep * 100))
        return out


def test_interpolate_timeline_passes_through_and_blends():
    # Source frames tagged with their index in the red channel.
    src = np.zeros((3, 8, 8, 3), np.uint8)
    for i in range(3):
        src[i, ..., 0] = i
    plan = [(0, 0.0), (0, 0.25), (1, 0.5), (2, 0.0)]
    rife = _FakeRife()

    out = interpolate_timeline(src, plan, rife)

    assert out.shape == (4, 8, 8, 3)
    assert int(out[0, 0, 0, 0]) == 0
    assert int(out[3, 0, 0, 0]) == 2
    assert rife.calls == [(0, 1, 0.25), (1, 2, 0.5)]


def test_weights_path_sits_under_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path))
    from server.config import get_settings

    get_settings.cache_clear()
    assert rife_weights_path() == tmp_path / "models" / "rife" / "flownet.pkl"
    get_settings.cache_clear()


def test_missing_weights_warning_is_actionable_chinese():
    # The warning is operator-facing in the admin console; pin the guidance it
    # gives so a silent downgrade never sneaks in.
    assert "RIFE" in SUB25_WARNING and "25" in SUB25_WARNING
