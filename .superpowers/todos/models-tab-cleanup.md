# Models tab reorg + reload bug + MiniMax config — DONE

## Branch
fix/musetalk-silent-static-video

## Summary
Three issues fixed in one branch.

### Issue 1: Models/Services reorg
- Portrait animators (LivePortrait, Wan 2.2) moved from `services` to `models` in `/api/admin/models`
- TTS row stays in `services` (remote service)
- Server: `_animator_service_row` → `_animator_model_row` with unified `ModelInfo` shape (engine + animator rows share flags `loaded`/`can_load`/`can_unload`)
- Added `LivePortrait.weights_dir` property so the row helper finds the weights directory
- Mock engine: added `kind: "drive_runtime"` + `label_zh` for unified shape
- UI: extended `ModelInfo` type with optional animator fields; `typeLabel` handles `portrait_animator`

### Issue 2: musetalk_runtime reload bug
- Root cause: `MuseTalkEngine.describe()` returned `loaded` + `can_unload` but no `can_load`
- UI used `disabled={!row.can_unload}` on load button → permanently disabled after unload
- Fix: added `can_load = has_load_method AND not loaded` to both engine + animator rows
- UI: load button now uses `disabled={!row.can_load}`
- New test `test_engine_can_reload_after_unload` pins the engine's load→unload→load round-trip

### Issue 3: MiniMax config page
- New helper `server/tts/config.py` with `read_config(db)` / `write_config(db, payload)` / `validate()` / `mask_api_key()`
- DB rows in the existing `settings` table (one per `minimax_*` key)
- Reads fresh on every synthesise call — bypasses `lru_cache` so console edits take effect immediately
- API: `GET /api/admin/tts/config` + `PUT /api/admin/tts/config`
- Pydantic `TtsConfigIn` with `extra='forbid'` rejects unknown fields at the boundary (422)
- Validation: `sample_rate` must be in {8000, 16000, 22050, 24000, 32000, 44100}; `base_url` must start with http(s)
- Masking: `api_key` never appears in GET/PUT response; only `api_key_masked` (last 4 or "(未设置)")
- UI: new `MiniMaxConfigCard` in Models.tsx — Form with API key (password), Group ID, Base URL, Model select, Voice ID, Sample rate select; 保存 + 测试 buttons

## Test results
- 46 new/updated tests pass
- 410 total passed in full suite (same as before minus new tests)
- 15 pre-existing torch-env failures (liveportrait/wan22/vram tests); none are regressions

## Verification
- Unit tests via .venv: pytest tests/test_tts_config.py tests/test_admin_tts_config.py tests/test_engine_models.py tests/test_musetalk_engine.py tests/test_admin_models.py tests/test_liveportrait_animator.py — all green
- Admin-ui: `npx tsc --noEmit` clean
- End-to-end (needs GPU host):
  - Open 系统 → 模型 → confirm LivePortrait / Wan 2.2 under 模型
  - Click 卸载 on `musetalk_runtime` → 加载 button re-enables, click reloads
  - MiniMax TTS 配置 card: change voice_id, save → toast confirms; test button round-trips