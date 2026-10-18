# Fix musetalk-silent static video — DONE

## Branch
fix/musetalk-silent-static-video (off main)

## Status
- All 12 tests in test_musetalk_silent_animator.py pass (3 new + 9 existing).
- 83 tests pass across musetalk_silent_animator / admin_animators / musetalk_engine / musetalk_pipeline.
- 15 pre-existing failures in liveportrait/wan22/vram tests are unrelated (torch/MuseTalk deps missing in .venv; same on main).

## Files changed
- server/avatar/musetalk_silent.py — replaced placeholder `_run_pipeline`; added `_read_wav_as_float32`; hoisted imports for `MuseTalkAvatarPipeline`, `AvatarPack`, `get_session_manager`.
- server/engine/musetalk_engine.py — added `drive_sync()` (submits `_Call` to worker queue, blocks on future, bypasses avatar cache via `_load_avatar` into a fresh `_SessionContext`).
- tests/test_musetalk_silent_animator.py — three new tests covering the real path; updated stub-fixture docstring; module-level `SimpleNamespace` import.

## Verification done
- RED → GREEN cycle for the three new tests, with intentional failures for the right reasons (placeholder doesn't call engine / doesn't raise on no-face).
- Full related test file suite green.
- Pre-existing failures confirmed unrelated (reproduced on stashed main).

## Verification NOT done (needs GPU host)
- End-to-end real inference: load musetalk runtime via admin UI, submit an i2v job, inspect the resulting mp4 for actual jaw/neck motion across frames.
- VRAM / cache state via `GET /api/admin/engines/musetalk` after i2v calls (should show `avatars_cached: 0`).

## Next
Commit on the fix branch.