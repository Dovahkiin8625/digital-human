# Remove musetalk-silent from i2v — DONE

## Branch
fix/musetalk-silent-static-video

## Why
MuseTalk is a lip-sync model. Feeding it silent audio doesn't produce "subtle closed-mouth motion" — it produces a static open-mouth frame, which is the wrong output for an idle/keep-alive clip. The "musetalk-silent" slot was a misdesign from the start.

## Files deleted
- `server/avatar/musetalk_silent.py`
- `tests/test_musetalk_silent_animator.py`

## Files modified (traces removed)
- `server/engine/musetalk_engine.py` — removed `drive_sync` (added only for this)
- `server/api/admin.py` — removed import, comments, and the `elif cid == "musetalk-silent"` branch
- `tests/test_admin_animators.py` — stripped musetalk-silent tests + references
- `tests/test_admin_models.py` — stripped musetalk-silent tests + references
- `tests/test_animator_loadable.py` — removed 4 lifecycle tests
- `tests/test_animator_vram.py` — removed meta test
- `admin-ui/src/pages/ActivatePortrait.tsx` — removed MuseTalk from help text
- `admin-ui/src/types.ts` — generic comment, no more "silently piggybacks" example
- `docs/superpowers/specs/2026-10-10-idle-video-playback-design.md` — updated YAGNI bullet

## What stays
- `MuseTalkEngine` (the streaming drive engine for live sessions) — unrelated to i2v
- `MuseTalkAvatarPipeline` (avatar pack preparation, used by streaming engine) — unrelated to i2v
- `/api/admin/i2v` endpoint — keeps working for LivePortrait + Wan 2.2
- The 形象活化 tab — LivePortrait / Wan 2.2 only now

## Test results
- Direct cleanup tests: 35/35 passed (test_admin_animators + test_admin_models)
- Broader suite: 8 pre-existing torch-env failures (liveportrait/wan22/vram), same on main and branch. No regressions introduced.
- 12 musetalk-silent tests deleted; corresponding source files gone.

## Recommended next backend for "subtle motion"
- LivePortrait + `subtle` driver (already implemented, already in ANIMATORS)
- Wan 2.2 (already implemented)