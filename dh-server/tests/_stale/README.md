# Stale tests

These files test features that no longer exist in this tree, so they cannot
even be imported:

| File | Missing |
|---|---|
| `test_admin_i2v_generate.py.stale` | `server.avatar.i2v` |
| `test_pipeline_i2v_branch.py.stale` | `server.avatar.i2v` |
| `test_overlay_compositor.py.stale` | `server.overlay` |
| `test_overlay_landmarks.py.stale` | `server.overlay` |
| `test_overlay_state.py.stale` | `server.overlay` |
| `test_render_session_overlay.py.stale` | `server.engine.mock_engine` (the mock moved to `tests/doubles.py` as `FakeEngine`) |

The `server.overlay` package was never in `main` — it lived on the
`feat/overlay-driving` branch. The `server.avatar.i2v` helper was folded into
the API layer, and the mock engine moved into the test doubles.

They are kept here, renamed `.py.stale` so pytest does not collect them,
rather than deleted: if one of those features comes back, the test is a
starting point, not something to write again.

The same fate used to be in store for `test_wan21_animator.py`,
`test_musetalk_silent_animator.py`, `test_control_ws.py`, `test_i2v_module.py`,
`test_admin_i2v_flag.py`, `test_live2d_kind.py`, `test_overlay_init.py`,
`test_overlay_prep.py`, `test_avatar_pipeline_overlay.py`,
`test_audio_loopback.py` and `test_lite_idle.py` — they imported modules
that no longer exist (`server.overlay`, `server.avatar.i2v`,
`server.avatar.wan21`, `server.avatar.musetalk_silent`, `server.lite`) or
called endpoints that were removed. They were deleted rather than kept:
they had been failing for so long that keeping them here would only have
been clutter, and the features they covered are gone for good.
