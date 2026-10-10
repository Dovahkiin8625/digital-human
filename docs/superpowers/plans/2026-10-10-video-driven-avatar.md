# 基于视频的形象制作 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「制作形象」支持上传短视频，视频作为动态驱动源（头部/身体沿用视频运动，嘴部由实时音频重生成），照片成为 N=1 的特例。

**Architecture:** 形象包统一为「N 帧循环」：磁盘存原始视频 + 小型 sidecar（latents/coords/masks/meta），引擎加载时用与流水线相同的解码函数重建底帧并按 `kept` 下标选取有效帧；渲染时按 `cycle_index(f, N)` 乒乓取帧，底图逐帧搬运到显存。照片走同一条路径（N=1）。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、numpy、Pillow、OpenCV（仅 GPU venv）、React 18 + Ant Design v5。

**Spec:** `docs/superpowers/specs/2026-10-10-video-driven-avatar-design.md`

## Global Constraints

- 运行环境：Windows 11 + Git Bash；命令均以仓库根为 CWD
- 默认测试 venv 为 `.venv`（torch-free）；GPU 测试在 `.venv-musetalk` 跑 `python -m pytest -m gpu`
- 默认测试命令：`python -m pytest`（`addopts` 已排除 `gpu`/`musetalk` 标记）
- 环境变量前缀 `DH_`；本次新增 `DH_AVATAR_MAX_VIDEO_SECONDS`、`DH_AVATAR_MAX_VIDEO_FRAMES`、`DH_AVATAR_CACHE_SIZE`
- 统一错误信封：`{"error": {"code": str, "message": str, "detail": any}}`；新增错误码 `invalid_video`、`video_too_long`（均 422）
- 画布固定 `720×1280`、输出帧率 `settings.fps=25`；`cv2` 只在 GPU venv 存在，凡用到 cv2 的模块必须在**函数内**惰性 import，保证 `.venv` 可导入
- 常量：`MAX_VIDEO_FRAMES=375`（15s×25fps）、`MIN_FRAMES=8`（视频最少有效帧）、`MIN_SIZE=256`（沿用）
- 循环为**乒乓**：`cycle_index(f,n)`，`n<=1 → 0`
- TDD：每个任务先写失败测试；每步完成后 `python -m pytest` 全绿再 commit
- Commit message 用 `feat/fix/test/docs/refactor` 前缀 + 说明性正文（无 trailer）

---

## File Structure

```
server/
├─ config.py                       # + avatar_max_video_seconds / avatar_max_video_frames / avatar_cache_size
├─ avatar/
│  ├─ video.py                     # 新增：probe_video / decode_frames / is_video_suffix / VideoError（cv2 惰性）
│  ├─ store.py                     # save_source_image → save_source（图片/视频分流 + 上限校验）
│  ├─ pipeline.py                  # MockAvatarPipeline 接受视频（取第 0 帧）
│  └─ musetalk_pipeline.py         # 统一 N 帧包格式；视频分支（解码→逐帧→丢无脸→MIN_FRAMES）
├─ engine/
│  └─ musetalk_engine.py           # cycle_index；_AvatarState 循环化；_render/_drive_sync；_load_avatar（新+旧）
└─ api/
   ├─ avatars.py                   # _create_avatar_rows 调 save_source（不再包 invalid_image）
   └─ admin.py                     # 同上（复用 _create_avatar_rows，无需改）

admin-ui/src/pages/Avatars.tsx     # 上传控件接受视频 + 文案 + 客户端时长预检

tests/
├─ test_video_decode.py            # 新增：probe/decode/is_video_suffix
├─ test_avatar_store_video.py      # 新增：save_source 视频校验
├─ test_musetalk_pipeline.py       # 改：统一包格式断言
├─ test_avatar_pipeline.py         # 改：save_source 改名；mock 视频
├─ test_avatars_api.py             # 改：视频上传生命周期 + 422 拒绝
├─ test_musetalk_engine.py         # 改：_AvatarState/_render 桩；+ cycle_index、逐帧渲染、新包加载
├─ test_admin_ui.py                # 改：源码守卫 accept="image/*,video/*"
└─ test_video_avatar_gpu.py        # 新增（gpu 标记）：真实权重端到端
```

---

### Task 1: 视频解码模块 `server/avatar/video.py`

**Files:**
- Create: `server/avatar/video.py`
- Modify: `server/config.py`
- Test: `tests/test_video_decode.py`

**Interfaces:**
- Consumes: `server.avatar.musetalk_pipeline.fit_to_canvas(img, width, height)`（惰性 import，避免导入环）
- Produces:
  - `VIDEO_SUFFIXES: set[str]`、`is_video_suffix(suffix: str) -> bool`
  - `class VideoError(Exception)`
  - `probe_video(path: Path) -> dict` → `{"fps": float, "frame_count": int, "duration": float}`，不可解码时抛 `VideoError`
  - `decode_frames(path: Path, target_fps: int, max_frames: int, canvas: tuple[int,int]) -> tuple[np.ndarray, float]` → `(frames[N,H,W,3] uint8 RGB, source_fps)`

- [ ] **Step 1: 写失败测试**

Create `tests/test_video_decode.py`:

```python
"""Contract tests for the video decoder.

cv2 lives only in the GPU venv, so `_cv2` is stubbed with a fake capture that
plays a list of solid-colour frames. A solid frame survives the letterbox
resize untouched, so the centre pixel identifies which source frame was picked.
"""
import numpy as np
import pytest

from server.avatar import video
from server.avatar.video import VideoError, decode_frames, is_video_suffix, probe_video

CANVAS = (720, 1280)


class _FakeCap:
    def __init__(self, frames, fps, opened=True):
        self._frames = frames
        self._fps = fps
        self._opened = opened
        self._i = 0

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == _FakeCV2.CAP_PROP_FPS:
            return self._fps
        if prop == _FakeCV2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frames, fps=25.0, opened=True):
        self.frames = frames
        self.fps = fps
        self.opened = opened

    def VideoCapture(self, path):
        # A fresh capture per call: probe() and decode_frames() each open their own.
        return _FakeCap(self.frames, self.fps, self.opened)


def _solid(n, fps=25.0):
    # frame i is a solid grey equal to i, at canvas size so letterbox is a no-op
    return [np.full((CANVAS[1], CANVAS[0], 3), i % 256, np.uint8) for i in range(n)], fps


def _install(monkeypatch, frames, fps=25.0, opened=True):
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(frames, fps, opened))


def test_is_video_suffix():
    assert is_video_suffix(".mp4")
    assert is_video_suffix(".MOV")
    assert not is_video_suffix(".png")
    assert not is_video_suffix("")


def test_probe_reports_fps_count_duration(monkeypatch):
    frames, fps = _solid(50, 25.0)
    _install(monkeypatch, frames, fps)
    info = probe_video("v.mp4")
    assert info["fps"] == 25.0
    assert info["frame_count"] == 50
    assert info["duration"] == pytest.approx(2.0)


def test_probe_raises_on_unopenable(monkeypatch):
    _install(monkeypatch, [], opened=False)
    with pytest.raises(VideoError):
        probe_video("v.mp4")


def test_decode_picks_the_nearest_source_frame_by_timestamp(monkeypatch):
    # 50 fps source, 25 fps target -> keep source frames 0,2,4,...
    frames, _ = _solid(50, 50.0)
    _install(monkeypatch, frames, 50.0)
    out, src_fps = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert src_fps == 50.0
    assert out.shape == (25, CANVAS[1], CANVAS[0], 3)
    assert out[0, 640, 360, 0] == 0
    assert out[1, 640, 360, 0] == 2
    assert out[2, 640, 360, 0] == 4


def test_decode_duplicates_frames_when_upsampling(monkeypatch):
    # 10 fps source, 25 fps target: several outputs share one source frame.
    frames, _ = _solid(10, 10.0)
    _install(monkeypatch, frames, 10.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert out.shape[0] == 25
    assert out[0, 640, 360, 0] == 0
    assert out[1, 640, 360, 0] == 0  # 1*10/25 = 0.4 -> rounds to source 0


def test_decode_caps_at_max_frames(monkeypatch):
    frames, _ = _solid(100, 25.0)
    _install(monkeypatch, frames, 25.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=10, canvas=CANVAS)
    assert out.shape[0] == 10


def test_decode_letterboxes_off_size_frames(monkeypatch):
    frames = [np.full((360, 640, 3), 200, np.uint8) for _ in range(25)]
    _install(monkeypatch, frames, 25.0)
    out, _ = decode_frames("v.mp4", target_fps=25, max_frames=375, canvas=CANVAS)
    assert out.shape[1:] == (CANVAS[1], CANVAS[0], 3)
    assert tuple(out[0, 5, 5]) == (16, 16, 24)  # letterbox bar
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_video_decode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.avatar.video'`

- [ ] **Step 3: 加配置项**

In `server/config.py`, add after `sample_rate: int = 48000`:

```python
    # Video-driven avatars. A driver video is decoded at the output fps and
    # capped so its frames can stay resident: 15 s at 25 fps is 375 frames.
    avatar_max_video_seconds: float = 15.0
    avatar_max_video_frames: int = 375
    # Video avatars keep ~1 GB of decoded frames resident each, so the engine's
    # per-avatar LRU is smaller than it was when every avatar was one frame.
    avatar_cache_size: int = 2
```

- [ ] **Step 4: 实现模块**

Create `server/avatar/video.py`:

```python
"""Video decoding for video-driven avatars.

Torch-free and cv2-lazy: `cv2` exists only in the GPU venv, so `_cv2()` is
imported at call time and the default `.venv` suite stubs it. `decode_frames`
is the single decoder shared by the preparation pipeline and the engine's load
path -- both call it with identical arguments so both see identical frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


class VideoError(Exception):
    pass


def is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in VIDEO_SUFFIXES


def _cv2():
    import cv2

    return cv2


def probe_video(path: Path) -> dict:
    """``{"fps", "frame_count", "duration"}``, or raise `VideoError`."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("cannot open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise VideoError("video has no readable frames")
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def _letterbox(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    # Lazily imported: musetalk_pipeline imports this module (lazily) too, and a
    # module-level import here would close the cycle.
    from server.avatar.musetalk_pipeline import fit_to_canvas

    return np.asarray(fit_to_canvas(Image.fromarray(rgb), width, height))


def decode_frames(
    path: Path, target_fps: int, max_frames: int, canvas: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Decode `path` to canvas-space RGB frames at `target_fps`, by timestamp.

    Returns ``(frames[N,H,W,3] uint8, source_fps)``. Picking the source frame
    nearest ``k/target_fps`` keeps motion at natural speed whatever the source
    fps is; where the source is slower than the target, frames repeat.
    """
    cv2 = _cv2()
    info = probe_video(path)
    src_fps = info["fps"]
    n_out = min(max_frames, int(round(info["duration"] * target_fps)))
    if n_out <= 0:
        raise VideoError("video is shorter than a single output frame")

    # source index -> the output frames that want it (a slow source feeds many)
    wanted: dict[int, list[int]] = {}
    for k in range(n_out):
        wanted.setdefault(int(round(k * src_fps / target_fps)), []).append(k)

    width, height = canvas
    frames = np.zeros((n_out, height, width, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("cannot open video")
        src_i = 0
        filled = 0
        while filled < n_out:
            ok, frame = cap.read()
            if not ok:
                break
            for k in wanted.get(src_i, ()):
                frames[k] = _letterbox(np.ascontiguousarray(frame[:, :, ::-1]), width, height)
                filled += 1
            src_i += 1
    finally:
        cap.release()
    if filled < n_out:
        # Probed N frames but decoded fewer: keep what actually came out.
        frames = frames[:filled]
    return frames, src_fps
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_video_decode.py -v`
Expected: PASS（7 passed）

- [ ] **Step 6: 对齐 spec 里的返回类型**

spec §6.1 把 `decode_frames` 的返回写成 `tuple[list[np.ndarray], float]`。实现返回 `np.ndarray [N,H,W,3]`：写入预分配数组（`frames[k] = ...`）而不是 N 次小分配，内存也保持连续。把 spec 该行改成：

```
decode_frames(path: Path, target_fps: int, max_frames: int, canvas: tuple[int, int])
    -> tuple[np.ndarray, float]   # (画布空间帧序列 [N,H,W,3] uint8, source_fps)
```

- [ ] **Step 7: 提交**

```bash
git add server/avatar/video.py server/config.py tests/test_video_decode.py docs/superpowers/specs/2026-10-10-video-driven-avatar-design.md
git commit -m "feat(avatar): video decoder with timestamp resampling and canvas letterbox"
```

---

### Task 2: `AvatarStore.save_source`（图片/视频分流 + 上限校验）

**Files:**
- Modify: `server/avatar/store.py`
- Modify: `server/api/avatars.py`（`_create_avatar_rows`）
- Modify: `tests/test_avatar_pipeline.py`（`test_save_source_image_validates_decodable`）
- Test: `tests/test_avatar_store_video.py`

**Interfaces:**
- Consumes: `server.avatar.video.probe_video`、`VideoError`、`is_video_suffix`（Task 1）
- Produces: `AvatarStore.save_source(avatar_id: str, data: bytes, suffix: str) -> Path`；失败抛 `server.core.errors.AppError`，code 为 `invalid_image` / `invalid_video` / `video_too_long`（均 422）

- [ ] **Step 1: 写失败测试**

Create `tests/test_avatar_store_video.py`:

```python
"""`save_source` routes by suffix: images decode via PIL, videos via a probe
that also enforces the duration/frame cap at upload time."""
import numpy as np
import pytest
from PIL import Image

from server.avatar.store import AvatarStore
from server.avatar.video import VideoError
from server.core.errors import AppError


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    from server.config import get_settings

    get_settings.cache_clear()
    return AvatarStore(tmp_path / "data")


def test_image_still_validated_by_pil(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    ok = tmp_path / "ok.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(ok)
    path = store.save_source("a1", ok.read_bytes(), ".png")
    assert path == store.avatar_dir("a1") / "source.png"

    with pytest.raises(AppError) as ei:
        store.save_source("a2", b"not an image", ".png")
    assert ei.value.code == "invalid_image"
    assert ei.value.status == 422


def test_video_accepts_a_probeable_file(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": 25.0, "frame_count": 50, "duration": 2.0},
    )
    path = store.save_source("v1", b"\x00" * 64, ".mp4")
    assert path == store.avatar_dir("v1") / "source.mp4"
    assert path.exists()


def test_video_rejects_undecodable(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)

    def boom(p):
        raise VideoError("cannot open video")

    monkeypatch.setattr("server.avatar.video.probe_video", boom)
    with pytest.raises(AppError) as ei:
        store.save_source("v2", b"\x00" * 64, ".mp4")
    assert ei.value.code == "invalid_video"
    assert ei.value.status == 422


def test_video_rejects_too_long(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": 25.0, "frame_count": 500, "duration": 20.0},
    )
    with pytest.raises(AppError) as ei:
        store.save_source("v3", b"\x00" * 64, ".mp4")
    assert ei.value.code == "video_too_long"
    assert ei.value.status == 422
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_avatar_store_video.py -v`
Expected: FAIL — `AttributeError: 'AvatarStore' object has no attribute 'save_source'`

- [ ] **Step 3: 实现 store 改动**

In `server/avatar/store.py`, replace the imports and `save_source_image`. Note that `from server.avatar.pipeline import PipelineError` goes away — `save_source` was its only user, and `load_pack`/`delete_assets` never raised it. `from server.avatar import video` (module, not names) is deliberate: the tests patch `server.avatar.video.probe_video`, which only takes effect if the call site looks the attribute up on the module at call time.

```python
import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from server.avatar import video
from server.config import get_settings
from server.core.errors import AppError
from server.db.models import Avatar
from server.engine.base import AvatarPack


class AvatarStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.data_dir / "avatars" / avatar_id

    def save_source(self, avatar_id: str, data: bytes, suffix: str) -> Path:
        """Persist the upload and reject anything unusable, before any job runs.

        An image is validated by decoding it; a video by probing it, which also
        enforces the duration/frame cap here rather than inside the background
        job -- a video that is too long should 422 at upload, not become a
        `failed` avatar ten seconds later.
        """
        d = self.avatar_dir(avatar_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"source{suffix or '.png'}"
        path.write_bytes(data)
        if video.is_video_suffix(suffix):
            self._validate_video(path)
        else:
            try:
                Image.open(path).verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise AppError("invalid_image", "source image is not a decodable image", 422) from exc
        return path

    def _validate_video(self, path: Path) -> None:
        settings = get_settings()
        try:
            info = video.probe_video(path)
        except video.VideoError as exc:
            raise AppError("invalid_video", str(exc), 422) from exc
        out_frames = int(info["duration"] * settings.fps)
        if info["duration"] > settings.avatar_max_video_seconds or out_frames > settings.avatar_max_video_frames:
            raise AppError(
                "video_too_long",
                f"video is {info['duration']:.1f}s (~{out_frames} frames); "
                f"limit is {settings.avatar_max_video_seconds:.0f}s / {settings.avatar_max_video_frames} frames",
                422,
            )
```

(Keep `load_pack` and `delete_assets` unchanged.)

- [ ] **Step 4: 让调用方不再包一层 `invalid_image`**

In `server/api/avatars.py`, `_create_avatar_rows` — replace the `try/except` block with a direct call:

```python
    avatar_id = uuid4().hex
    store = _store()
    source_path = store.save_source(avatar_id, data, suffix)

    avatar = Avatar(
```

(`save_source` now raises `AppError` with the right code itself, so the wrapper would mislabel a video error as `invalid_image`.)

- [ ] **Step 5: 更新既有测试**

In `tests/test_avatar_pipeline.py`, replace `test_save_source_image_validates_decodable`:

```python
def test_save_source_validates_decodable(tmp_path):
    store = AvatarStore(tmp_path / "data")
    ok = tmp_path / "ok.png"
    Image.new("RGB", (300, 300), (1, 2, 3)).save(ok)

    path = store.save_source("a1", ok.read_bytes(), ".png")
    assert path == store.avatar_dir("a1") / "source.png"
    assert path.exists()

    from server.core.errors import AppError

    with pytest.raises(AppError) as ei:
        store.save_source("a2", b"not an image", ".png")
    assert ei.value.code == "invalid_image"
```

- [ ] **Step 6: 运行确认通过**

Run: `python -m pytest tests/test_avatar_store_video.py tests/test_avatar_pipeline.py tests/test_avatars_api.py -v`
Expected: PASS（全部通过；`test_avatars_api.py` 未改动的用例仍绿）

- [ ] **Step 7: 提交**

```bash
git add server/avatar/store.py server/api/avatars.py tests/test_avatar_store_video.py tests/test_avatar_pipeline.py
git commit -m "feat(avatar): save_source routes images and videos, capping video at upload"
```

---

### Task 3: MuseTalk 流水线改为统一 N 帧包格式（图片路径）

**Files:**
- Modify: `server/avatar/musetalk_pipeline.py`
- Modify: `tests/test_musetalk_pipeline.py`

**Interfaces:**
- Produces: 统一包布局 —— `normalized.png`、`musetalk/latents.pt`（`[N,8,32,32]`）、`musetalk/coords.json`（`[{"bbox","mask_crop_box"}] × N`）、`musetalk/masks/NNNNNNNN.png × N`、`musetalk/meta.json`；`pack.json` 的 `musetalk` 块含 `mode/source/frame_count/fps/source_fps/kept/coords/masks/...`
- Produces: `_save_latents(parts, path)` 新签名（`parts` 为 `[1,8,32,32]` 列表）
- Produces: `MIN_FRAMES = 8`

本任务**只改图片路径**（N=1）。视频分支在 Task 4。

- [ ] **Step 1: 改写契约测试**

In `tests/test_musetalk_pipeline.py`, replace `SIDECARS` and `test_pack_contract_is_complete`, and fix the latents stub.

Replace the module-level `SIDECARS = [...]` line with:

```python
SIDECARS = ["latents.pt", "coords.json", "meta.json"]
```

In the `wired` fixture, replace the `fake_save_latents` body:

```python
    def fake_save_latents(parts, path):
        arr = np.concatenate([np.asarray(p) for p in parts], axis=0)
        written["shape"] = tuple(arr.shape)
        with open(path, "wb") as fh:
            np.save(fh, arr)
```

Replace `test_pack_contract_is_complete`:

```python
def test_pack_contract_is_complete(wired):
    pipeline, _, src, out, written = wired
    stages = []

    cfg = pipeline.process("a1", src, out, lambda s, p: stages.append((s, p)))

    assert stages == [("detect_face", 10), ("align", 40), ("normalize", 70), ("build_pack", 100)]

    sidecar = out / "musetalk"
    for name in SIDECARS:
        assert (sidecar / name).exists(), f"missing sidecar {name}"
    assert (sidecar / "masks" / "00000000.png").exists()

    assert written["shape"] == (1, 8, 32, 32), "one image frame -> latents [1,8,32,32]"

    with Image.open(out / "normalized.png") as img:
        assert img.size == (720, 1280)

    coords = json.loads((sidecar / "coords.json").read_text())
    assert len(coords) == 1
    assert coords[0]["bbox"] == [100, 200, 300, 400]
    assert coords[0]["mask_crop_box"] == [10, 20, 310, 320]

    meta = json.loads((sidecar / "meta.json").read_text())
    assert meta["version"] == "1.5"
    assert meta["crop"] == FACE_CROP
    assert meta["mode"] == "image"
    assert meta["frame_count"] == 1
    assert meta["kept"] == [0]

    # pack.json carries only the small "musetalk" block, and it must be JSON-clean
    assert list(cfg) == ["musetalk"]
    assert json.loads(json.dumps(cfg)) == cfg
    assert cfg["musetalk"]["coords"] == "musetalk/coords.json"
    assert cfg["musetalk"]["masks"] == "musetalk/masks"
    assert cfg["musetalk"]["bbox"] == [100, 200, 300, 400]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_musetalk_pipeline.py -v`
Expected: FAIL — `KeyError: 'coords.json'` / missing `musetalk/coords.json`

- [ ] **Step 3: 实现统一写入**

In `server/avatar/musetalk_pipeline.py`.

First, replace the module docstring's layout block (lines 3-19), which still describes `face_crop.png` / `bbox.json` / `landmarks.json` / `mask.png`:

```python
"""Real MuseTalk 1.5 avatar preparation -- the production path.

Turns one uploaded photo or one short driver video into the artifacts the engine
needs. A photo is the N=1 case of a frame cycle, so both modes write the same
layout:

    <avatar_dir>/normalized.png          720x1280 base frame, taken from the first
                                         kept frame (idle keepalive, preview, and
                                         the shape the old single-frame packs had)
    <avatar_dir>/pack.json               small scalars/paths only (no tensors)
    <avatar_dir>/musetalk/
        latents.pt      [N,8,32,32]      VAE latents, one row per kept frame
        coords.json     [N]              face box + blend crop box per kept frame
        masks/NNNNNNNN.png               bisenet blend mask per kept frame, sized
                                         to that frame's own mask_crop_box
        meta.json       model version / crop / detector / fps / mode / kept

`kept` is the list of decoded-sequence indices the N rows came from: a video
drops frames with no detectable face, so row i of `coords.json` is decoded frame
`kept[i]`, and the engine re-decodes the same sequence and selects those indices.

Everything torch-shaped is imported lazily inside `MuseTalkPreparer`, so this
module stays importable (and the default test suite stays torch-free) even
though it is only ever *used* in the GPU venv.
"""
```

Add near the other constants:

```python
#: A video needs at least this many face-bearing frames to be usable as a driver.
MIN_FRAMES = 8
```

Replace `_save_latents` with the list-taking seam:

```python
def _save_latents(parts, path: Path) -> None:
    """Seam for the torch-free contract test.

    `parts` is a list of per-frame ``[1,8,32,32]`` latents; they are stacked to
    ``[N,8,32,32]`` and saved as one tensor. Tests substitute a numpy writer.
    """
    _, torch = _cv2_and_torch()
    torch.save(torch.cat(list(parts), dim=0).detach().cpu(), path)
```

Replace `MuseTalkAvatarPipeline.process` with the unified, frame-list-driven version (video branch lands in Task 4, so this task's `_video_frames` is a stub that is never reached yet):

```python
    def process(
        self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]
    ) -> dict:
        settings = get_settings()
        source_path = Path(source_path)
        mode = "video" if is_video_suffix(source_path.suffix) else "image"

        progress("detect_face", 10)
        if mode == "video":
            frames, source_fps = self._video_frames(source_path, settings)
        else:
            frames, source_fps = self._image_frames(source_path, settings)

        prep = self._get_preparer()
        parts, coords, masks, kept = [], [], [], []
        for i, bgr in enumerate(frames):
            try:
                out = prep(bgr)
            except PipelineError:
                if mode == "image":
                    raise  # an image with no face is a plain error, not a skip
                continue
            kept.append(i)
            parts.append(out["latents"])
            coords.append({"bbox": out["bbox"], "mask_crop_box": out["mask_crop_box"]})
            masks.append(out["mask"])

        if not parts:
            raise PipelineError("no face detected in the uploaded image")
        if mode == "video" and len(parts) < MIN_FRAMES:
            raise PipelineError(f"no usable face frames in video: found {len(parts)}, need {MIN_FRAMES}")

        progress("align", 40)
        sidecar = out_dir / SUBDIR
        sidecar.mkdir(parents=True, exist_ok=True)
        masks_dir = sidecar / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        _save_latents(parts, sidecar / "latents.pt")
        (sidecar / "coords.json").write_text(json.dumps(coords))
        for i, mask in enumerate(masks):
            Image.fromarray(mask).save(masks_dir / f"{i:08d}.png")

        progress("normalize", 70)
        # Frame 0 of the *kept* frames: the resting canvas, preview and back-compat.
        Image.fromarray(frames[kept[0]][:, :, ::-1]).save(out_dir / "normalized.png")

        progress("build_pack", 100)
        (sidecar / "meta.json").write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "crop": FACE_CROP,
                    "detector": DETECTOR,
                    "fps": settings.fps,
                    "source_fps": source_fps,
                    "frame_count": len(parts),
                    "mode": mode,
                    "kept": kept,
                }
            )
        )
        return {
            "musetalk": {
                "mode": mode,
                "dir": SUBDIR,
                "version": VERSION,
                "crop": FACE_CROP,
                "detector": DETECTOR,
                "frame_count": len(parts),
                "fps": settings.fps,
                "source_fps": source_fps,
                "base_image": "normalized.png",
                "source": source_path.name,
                "latents": f"{SUBDIR}/latents.pt",
                "coords": f"{SUBDIR}/coords.json",
                "masks": f"{SUBDIR}/masks",
                "bbox": coords[0]["bbox"],
                "mask_crop_box": coords[0]["mask_crop_box"],
            }
        }

    def _image_frames(self, source_path: Path, settings) -> tuple[list[np.ndarray], float]:
        img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(
                f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}"
            )
        canvas = fit_to_canvas(img, settings.frame_width, settings.frame_height)
        return [np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])], float(settings.fps)

    def _video_frames(self, source_path: Path, settings) -> tuple[list[np.ndarray], float]:
        raise NotImplementedError  # Task 4
```

Add the import at the top of the module:

```python
from server.avatar.video import is_video_suffix
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_musetalk_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add server/avatar/musetalk_pipeline.py tests/test_musetalk_pipeline.py
git commit -m "refactor(avatar): unify the musetalk pack to an N-frame layout"
```

---

### Task 4: MuseTalk 流水线的视频分支

**Files:**
- Modify: `server/avatar/musetalk_pipeline.py`（`_video_frames`）
- Test: `tests/test_video_pipeline.py`

**Interfaces:**
- Consumes: `server.avatar.video.decode_frames`（Task 1）、`MIN_FRAMES`（Task 3）
- Produces: 视频形象包（同 Task 3 布局，`mode="video"`，`kept` 为解码序列下标）

- [ ] **Step 1: 写失败测试**

Create `tests/test_video_pipeline.py`:

```python
"""The video branch of the musetalk pipeline: decode, drop face-less frames,
record which decoded frames survived, and refuse a video with too few faces."""
import json

import numpy as np
import pytest
from PIL import Image

from server.avatar import musetalk_pipeline as mp
from server.avatar.musetalk_pipeline import FACE_CROP, MuseTalkAvatarPipeline
from server.avatar.pipeline import PipelineError

CANVAS = (720, 1280)


class StubPreparer:
    """Detects a face in every frame except the ones whose centre is black."""

    def __init__(self, blank=()):
        self.blank = set(blank)
        self.seen = 0

    def __call__(self, bgr):
        i = self.seen
        self.seen += 1
        if i in self.blank:
            raise PipelineError("no face detected in the uploaded image")
        return {
            "bbox": [100, 200, 300, 400],
            "landmarks": np.zeros((468, 2), dtype=np.float32),
            "face_crop_bgr": np.zeros((FACE_CROP, FACE_CROP, 3), dtype=np.uint8),
            "latents": np.zeros((1, 8, 32, 32), dtype=np.float32),
            "mask": np.full((300, 300), 255, dtype=np.uint8),
            "mask_crop_box": [10, 20, 310, 320],
        }


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("DH_DATA_DIR", str(tmp_path / "data"))
    from server.config import get_settings

    get_settings.cache_clear()
    written = {}

    def fake_save_latents(parts, path):
        arr = np.concatenate([np.asarray(p) for p in parts], axis=0)
        written["shape"] = tuple(arr.shape)
        with open(path, "wb") as fh:
            np.save(fh, arr)

    monkeypatch.setattr(mp, "_save_latents", fake_save_latents)
    return written, tmp_path


def _video(monkeypatch, n_frames):
    def fake_decode(path, target_fps, max_frames, canvas):
        frames = np.full((n_frames, canvas[1], canvas[0], 3), 120, np.uint8)
        return frames, 30.0

    monkeypatch.setattr("server.avatar.video.decode_frames", fake_decode)


def test_video_pack_records_kept_indices(wired, monkeypatch):
    written, tmp_path = wired
    _video(monkeypatch, 12)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)

    # frames 3 and 7 have no face and must be dropped
    prep = StubPreparer(blank={3, 7})
    cfg = MuseTalkAvatarPipeline(preparer=prep).process("v1", src, tmp_path / "out", lambda s, p: None)

    sidecar = tmp_path / "out" / "musetalk"
    coords = json.loads((sidecar / "coords.json").read_text())
    meta = json.loads((sidecar / "meta.json").read_text())

    assert meta["mode"] == "video"
    assert meta["frame_count"] == 10
    assert meta["kept"] == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    assert len(coords) == 10
    assert written["shape"] == (10, 8, 32, 32)
    assert len(list((sidecar / "masks").glob("*.png"))) == 10
    assert cfg["musetalk"]["source"] == "clip.mp4"
    assert cfg["musetalk"]["frame_count"] == 10


def test_video_with_too_few_faces_is_rejected(wired, monkeypatch):
    _, tmp_path = wired
    _video(monkeypatch, 12)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x00" * 16)

    prep = StubPreparer(blank=set(range(5, 12)))  # only 5 usable frames < MIN_FRAMES
    with pytest.raises(PipelineError, match="no usable face frames"):
        MuseTalkAvatarPipeline(preparer=prep).process("v2", src, tmp_path / "out", lambda s, p: None)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_video_pipeline.py -v`
Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: 实现视频分支**

In `server/avatar/musetalk_pipeline.py`, replace the `_video_frames` stub:

```python
    def _video_frames(self, source_path: Path, settings) -> tuple[list[np.ndarray], float]:
        from server.avatar.video import decode_frames

        frames, source_fps = decode_frames(
            source_path,
            target_fps=settings.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if frames.shape[0] == 0:
            raise PipelineError("video has no decodable frames")
        # The preparer expects BGR; decode_frames returns the canvas in RGB.
        return [np.ascontiguousarray(f[:, :, ::-1]) for f in frames], source_fps
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_video_pipeline.py tests/test_musetalk_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add server/avatar/musetalk_pipeline.py tests/test_video_pipeline.py
git commit -m "feat(avatar): video branch in the musetalk pipeline, recording kept frames"
```

---

### Task 5: Mock 流水线接受视频 + API 端到端

**Files:**
- Modify: `server/avatar/pipeline.py`（`MockAvatarPipeline.process`）
- Test: `tests/test_avatars_api.py`（新增视频用例）

**Interfaces:**
- Consumes: `server.avatar.video.decode_frames`、`is_video_suffix`
- Produces: mock 流水线对视频取第 0 帧，产物与图片一致

- [ ] **Step 1: 写失败测试**

Append to `tests/test_avatars_api.py`:

```python
def _stub_video(monkeypatch, duration=2.0, fps=25.0):
    # Both call sites look the names up on the module at call time (store does
    # `video.probe_video`, the mock pipeline `video.decode_frames`), which is
    # what makes patching `server.avatar.video` sufficient for both.
    monkeypatch.setattr(
        "server.avatar.video.probe_video",
        lambda p: {"fps": fps, "frame_count": int(duration * fps), "duration": duration},
    )
    import numpy as np

    monkeypatch.setattr(
        "server.avatar.video.decode_frames",
        lambda p, target_fps, max_frames, canvas: (
            np.full((1, canvas[1], canvas[0], 3), 128, np.uint8),
            fps,
        ),
    )


def test_avatar_video_lifecycle(client, monkeypatch):
    _stub_video(monkeypatch)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Video Alice"},
        files={"file": ("alice.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 201, resp.text
    avatar_id = resp.json()["avatar_id"]

    detail = client.get(f"/api/v1/avatars/{avatar_id}", headers=headers).json()
    assert detail["status"] == "ready"


def test_avatar_video_too_long_is_422(client, monkeypatch):
    _stub_video(monkeypatch, duration=20.0)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Too long"},
        files={"file": ("long.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "video_too_long"


def test_avatar_undecodable_video_is_422(client, monkeypatch):
    from server.avatar.video import VideoError

    def boom(p):
        raise VideoError("cannot open video")

    monkeypatch.setattr("server.avatar.video.probe_video", boom)
    headers = _seed_tenant(client)
    resp = client.post(
        "/api/v1/avatars",
        headers=headers,
        data={"name": "Bad"},
        files={"file": ("bad.mp4", b"\x00" * 64, "video/mp4")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_video"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_avatars_api.py -k video -v`
Expected: FAIL — `test_avatar_video_lifecycle` 的 avatar 停在 `failed`（mock 用 PIL 打开 mp4 失败）

- [ ] **Step 3: 实现 mock 视频分支**

In `server/avatar/pipeline.py`, add the import and replace the head of `MockAvatarPipeline.process`:

```python
from server.avatar import video
```

Module-style on purpose: `server.avatar.video` imports only numpy at module level (cv2 is lazy inside `_cv2()`), so this adds no GPU dependency to the torch-free `.venv`, and looking the names up at call time is what lets the tests patch `server.avatar.video.decode_frames`.

```python
    def process(self, avatar_id: str, source_path: Path, out_dir: Path, progress: Callable[[str, int], None]) -> dict:
        progress("detect_face", 10)
        source_path = Path(source_path)
        if video.is_video_suffix(source_path.suffix):
            settings = get_settings()
            frames, _ = video.decode_frames(
                source_path,
                target_fps=settings.fps,
                max_frames=1,
                canvas=(settings.frame_width, settings.frame_height),
            )
            img = Image.fromarray(frames[0])
        else:
            img = Image.open(source_path).convert("RGB")
        if img.width < self.MIN_SIZE or img.height < self.MIN_SIZE:
            raise PipelineError(f"image too small: {img.width}x{img.height}, min {self.MIN_SIZE}x{self.MIN_SIZE}")
        progress("align", 40)
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        cropped = img.crop((left, top, left + side, top + side))
        progress("normalize", 70)
        out_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (720, 1280), (16, 16, 24))
        resized = cropped.resize((720, 720))
        canvas.paste(resized, (0, 280))
        canvas.save(out_dir / "normalized.png")
        progress("build_pack", 100)
        return {"mouth_scale": 1.0, "expression_intensity": 1.0, "crop": {"mode": "center"}}
```

(Keep the rest of the class and `create_pipeline`/`run_avatar_job` unchanged.)

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_avatars_api.py tests/test_avatar_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add server/avatar/pipeline.py tests/test_avatars_api.py
git commit -m "feat(avatar): mock pipeline and tenant API accept a video upload"
```

---

### Task 6: 引擎循环渲染（`cycle_index`、`_AvatarState`、`_render`）

**Files:**
- Modify: `server/engine/musetalk_engine.py`
- Modify: `tests/test_musetalk_engine.py`（桩更新 + 新增循环用例）

**Interfaces:**
- Produces: `cycle_index(frame: int, n: int) -> int`（模块级）
- Produces: `_AvatarState(frames, latents, bboxes, masks, mask_boxes)`，`cycle_len` 为属性；删除 `base_t`/`mask_t`
- Produces: `_render(rt, avatar, idxs: list[int], prompts)` —— `idxs[j]` 是第 j 个输出帧的循环下标
- Consumes: `settings.avatar_cache_size`（Task 1）

本任务**不含**新包加载（Task 7）；本任务把 `_load_avatar` 改为产出新的 `_AvatarState`（仍只走旧格式）。

- [ ] **Step 1: 写失败测试**

In `tests/test_musetalk_engine.py`, add a new section near the end:

```python
# -- frame cycle --------------------------------------------------------

def test_cycle_index_ping_pongs():
    assert me.cycle_index(0, 1) == 0
    assert me.cycle_index(99, 1) == 0
    for f, want in [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (5, 1), (6, 2)]:
        assert me.cycle_index(f, 3) == want, f
    # n=2 has no duplicate end frame: 0,1,0,1,...
    assert [me.cycle_index(f, 2) for f in range(6)] == [0, 1, 0, 1, 0, 1]


def test_render_uses_a_different_base_per_output_frame(tmp_path, monkeypatch):
    """Two output frames must blend onto two different base frames."""
    import torch

    eng = MuseTalkEngine(models_dir=tmp_path / "m")

    seen_bases = []

    def spy_blend(base, face, bbox, mask, crop_box):
        seen_bases.append(int(base[0, 0, 0]))  # base is CHW; corner pixel tags it
        return base

    monkeypatch.setattr(me, "_blend_face_gpu", spy_blend)

    class _Model:
        def __call__(self, latents, timesteps, encoder_hidden_states=None):
            class _Out:
                sample = latents

            return _Out()

    class _UNet:
        model = _Model()

    class _RT:
        device = "cpu"
        weight_dtype = torch.float32
        timesteps = torch.tensor([0])
        unet = _UNet()

        def pe(self, wb):
            return wb

    # `_render` reaches the model as `rt.unet.model(...)`, so the stub needs the
    # `unet` level -- and `_decode_gpu` is stubbed out because the fake UNet
    # returns latents, not pixels.
    monkeypatch.setattr(me, "_decode_gpu", lambda rt, pred: pred)

    frames = np.zeros((2, 8, 8, 3), np.uint8)
    frames[0, :, :, :] = 10
    frames[1, :, :, :] = 20
    avatar = me._AvatarState(
        frames=frames,
        latents=torch.zeros((2, 8, 32, 32)),
        bboxes=[[0, 0, 8, 8], [0, 0, 8, 8]],
        masks=[np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8)],
        mask_boxes=[[0, 0, 8, 8], [0, 0, 8, 8]],
    )
    prompts = _FakePrompts(np.zeros((2, 50, 384), np.float32))

    eng._render(_RT(), avatar, [0, 1], prompts)

    assert seen_bases == [10, 20]
```

Update the existing stubs so the suite runs against the new shape:

In the `engine` fixture, replace the `_render` monkeypatch and the `_AvatarState(...)` construction:

```python
    monkeypatch.setattr(
        eng,
        "_render",
        lambda rt, avatar, idxs, prompts: [np.zeros((1280, 720, 3), np.uint8)] * len(idxs),
    )
    monkeypatch.setattr(
        eng,
        "_load_avatar",
        lambda rt, pack: me._AvatarState(
            frames=np.zeros((1, 1280, 720, 3), np.uint8),
            latents=None,
            bboxes=[[100, 200, 300, 400]],
            masks=[np.zeros((272, 272), np.uint8)],
            mask_boxes=[[50, 150, 350, 450]],
        ),
    )
```

In `test_concurrent_sessions_share_a_render_batch`, change the spy signature:

```python
    def spy(rt, avatar, idxs, prompts):
        time.sleep(0.005)  # stand in for the ~60 ms of real render work
        calls.append(int(prompts.shape[0]))
        return real_render(rt, avatar, idxs, prompts)
```

In `test_drive_does_not_block_the_event_loop`, replace its `_load_avatar` stub with the same shape:

```python
    monkeypatch.setattr(
        eng,
        "_load_avatar",
        lambda rt, pack: me._AvatarState(
            frames=np.zeros((1, 1280, 720, 3), np.uint8),
            latents=None,
            bboxes=[[100, 200, 300, 400]],
            masks=[np.zeros((272, 272), np.uint8)],
            mask_boxes=[[50, 150, 350, 450]],
        ),
    )
```

In `test_load_avatar_reads_sidecars`, replace the final four assertions:

```python
    assert state.bboxes[0] == [100, 200, 300, 400]
    assert state.mask_boxes[0] == [50, 150, 350, 450]
    assert state.frames[0].shape == (1280, 720, 3)
    assert state.masks[0].shape == (272, 272)
    assert state.cycle_len == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_musetalk_engine.py -v`
Expected: FAIL — `AttributeError: module 'server.engine.musetalk_engine' has no attribute 'cycle_index'`

- [ ] **Step 3: 实现循环渲染**

In `server/engine/musetalk_engine.py`.

First, update the module docstring's model-state block (lines 8-13), which still says "per-avatar cache -- latents/mask/base frame, LRU by avatar_id":

```
  * global weights   -- VAE/UNet/whisper, loaded once, lock-protected, warmed
                        at startup (cold start is ~34 s of loading + kernel
                        autotune, which must not land on the first session)
  * per-avatar cache -- base frames + per-frame latents/bbox/mask, LRU by
                        avatar_id. A video avatar holds N frames; an image
                        avatar is the N=1 case.
```

Add the module-level helper (after the `_settle` helper):

```python
def cycle_index(frame: int, n: int) -> int:
    """Ping-pong index into an n-frame cycle: 0..n-1, n-1..1, 0..

    MuseTalk's own `frame_list_cycle = frames + frames[::-1]`, without storing
    the frames twice. n<=1 collapses to 0 so an image avatar needs no special case.
    """
    if n <= 1:
        return 0
    period = 2 * n - 2
    k = frame % period
    return k if k < n else period - k
```

Replace the `AVATAR_CACHE_SIZE = 4` constant's use: delete the module constant and read the setting in `__init__`:

```python
        self.cache_size = int(settings.avatar_cache_size)
```

In `_ensure_avatar`, replace `AVATAR_CACHE_SIZE` with `self.cache_size`:

```python
        while len(self._avatars) > self.cache_size:
            self._avatars.popitem(last=False)
```

Replace `_AvatarState`:

```python
@dataclass
class _AvatarState:
    """One avatar's prepared artifacts, resolved from pack.prep.

    A video avatar carries N base frames and their per-frame sidecars; an image
    avatar is the N=1 case. `frames` stays on the CPU -- a 15 s avatar is ~1 GB
    -- and only the frames a batch needs are moved to the device.
    """

    frames: np.ndarray  # [N,H,W,3] uint8, canvas space, sink's RGB order
    latents: object  # torch.Tensor [N,8,32,32] on device, weight dtype
    bboxes: list[list[int]]  # length N
    masks: list[np.ndarray]  # length N, each sized to its frame's mask_crop_box
    mask_boxes: list[list[int]]  # length N

    @property
    def cycle_len(self) -> int:
        return len(self.bboxes)
```

Replace `_device_assets` with per-frame converters:

```python
    def _to_device(self, rt: _Runtime, rgb: np.ndarray):
        import torch

        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(rt.device)

    def _mask_to_device(self, rt: _Runtime, mask: np.ndarray):
        import torch

        return (
            torch.from_numpy(mask).to(torch.float32).div_(255.0).unsqueeze(0).to(rt.device)
        )
```

Replace `_render`:

```python
    def _render(self, rt: _Runtime, avatar: _AvatarState, idxs: list[int], prompts) -> list[np.ndarray]:
        import torch
        import torch.nn.functional as F

        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, prompts.shape[0], BATCH_SIZE):
                wb = prompts[i : i + BATCH_SIZE].to(rt.device, dtype=rt.weight_dtype)
                lb = avatar.latents[[idxs[i + j] for j in range(wb.shape[0])]]
                pred = rt.unet.model(lb, rt.timesteps, encoder_hidden_states=rt.pe(wb)).sample
                recon = _decode_gpu(rt, pred)
                for j in range(recon.shape[0]):
                    c = idxs[i + j]
                    bbox = avatar.bboxes[c]
                    x1, y1, x2, y2 = bbox
                    face = (
                        F.interpolate(
                            recon[j : j + 1].to(torch.float32),
                            size=(y2 - y1, x2 - x1),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .round()
                        .clamp_(0.0, 255.0)
                        .to(torch.uint8)[0]
                    )
                    frame = _blend_face_gpu(
                        self._to_device(rt, avatar.frames[c]),
                        face,
                        bbox,
                        self._mask_to_device(rt, avatar.masks[c]),
                        avatar.mask_boxes[c],
                    )
                    out.append(frame.permute(1, 2, 0).contiguous().cpu().numpy())
        return out
```

In `_drive_sync`, replace the group loop body that computes prompts and calls `_render`:

```python
            for idxs in groups.values():
                avatar = work[idxs[0]][1]
                cycle_idxs: list[int] = []
                for k in idxs:
                    _, _, first, pr = work[k]
                    cycle_idxs += [cycle_index(first + j, avatar.cycle_len) for j in range(int(pr.shape[0]))]
                prompts = _cat_prompts([work[k][3] for k in idxs])
                images = self._render(rt, avatar, cycle_idxs, prompts)
                pos = 0
                for k in idxs:
                    i, _, first, pr = work[k]
                    n = int(pr.shape[0])
                    out[i] = [
                        DriveFrame(pts=(first + j) / self.fps, image=images[pos + j])
                        for j in range(n)
                    ]
                    pos += n
```

Finally, make `_load_avatar` produce the new shape (still only the legacy layout, which Task 7 extends):

```python
    def _load_avatar(self, rt: _Runtime, pack: AvatarPack) -> _AvatarState:
        from PIL import Image

        prep = pack.prep or {}
        if not prep.get("latents") or not (prep.get("coords") or prep.get("mask")):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} has no musetalk pack; prepare it with "
                "DH_AVATAR_PIPELINE=musetalk"
            )
        root = Path(pack.root)
        base_rgb = np.array(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        return _AvatarState(
            frames=base_rgb[None, ...],
            latents=latents,
            bboxes=[[int(v) for v in prep["bbox"]]],
            masks=[np.array(Image.open(root / prep["mask"]).convert("L"))],
            mask_boxes=[[int(v) for v in prep["mask_crop_box"]]],
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_musetalk_engine.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add server/engine/musetalk_engine.py tests/test_musetalk_engine.py
git commit -m "feat(engine): render onto a per-frame ping-pong cycle instead of one base"
```

---

### Task 7: 引擎加载视频包（新布局 + 旧格式回退）

**Files:**
- Modify: `server/engine/musetalk_engine.py`（`_load_avatar`）
- Test: `tests/test_musetalk_engine.py`（新增视频包加载用例）

**Interfaces:**
- Consumes: `meta.json` 的 `kept`/`frame_count`、`pack.prep["source"|"coords"|"masks"|"latents"]`
- Produces: `_load_video_avatar(rt, pack, root, prep) -> _AvatarState`，帧数/下标不一致时抛 `EngineUnavailable`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_musetalk_engine.py`:

```python
# -- video pack loading -------------------------------------------------

def _video_pack(tmp_path, monkeypatch, n_kept=3, decoded=4, kept=None):
    """A pack.json block in the new video layout, plus the sidecars it names."""
    import json

    root = tmp_path
    sidecar = root / "musetalk"
    (sidecar / "masks").mkdir(parents=True, exist_ok=True)
    (root / "source.mp4").write_bytes(b"\x00")
    kept = kept if kept is not None else list(range(n_kept))
    coords = [{"bbox": [0, 0, 8, 8], "mask_crop_box": [0, 0, 8, 8]} for _ in kept]
    (sidecar / "coords.json").write_text(json.dumps(coords))
    (sidecar / "meta.json").write_text(
        json.dumps({"frame_count": len(kept), "kept": kept, "fps": 25, "source_fps": 25.0, "mode": "video"})
    )
    for i in range(len(kept)):
        Image.new("L", (8, 8), 255).save(sidecar / "masks" / f"{i:08d}.png")
    with open(sidecar / "latents.pt", "wb") as fh:
        np.save(fh, np.zeros((len(kept), 8, 32, 32), np.float32))

    class _Loaded:
        """Stands in for a torch tensor: `.shape` and a no-op `.to()`.

        `_Moved` (used by the legacy test) has no shape, and the video loader
        validates `latents.shape[0]` against `kept` -- so this stub reads the
        file the test actually wrote, which is what makes that check meaningful.
        """

        def __init__(self, path):
            with open(path, "rb") as fh:
                self.arr = np.load(fh)

        @property
        def shape(self):
            return self.arr.shape

        def to(self, *args, **kwargs):
            return self

    monkeypatch.setattr(me, "_load_latents", lambda path: _Loaded(path))
    decoded_frames = np.zeros((decoded, 16, 16, 3), np.uint8)
    for i in range(decoded):
        decoded_frames[i] = i + 1
    monkeypatch.setattr(
        "server.avatar.video.decode_frames",
        lambda p, target_fps, max_frames, canvas: (decoded_frames, 25.0),
    )
    prep = {
        "mode": "video",
        "dir": "musetalk",
        "source": "source.mp4",
        "base_image": "normalized.png",
        "latents": "musetalk/latents.pt",
        "coords": "musetalk/coords.json",
        "masks": "musetalk/masks",
        "bbox": [0, 0, 8, 8],
        "mask_crop_box": [0, 0, 8, 8],
    }
    return _pack(root, prep=prep)


def test_load_video_pack_selects_kept_frames(tmp_path, monkeypatch):
    pack = _video_pack(tmp_path, monkeypatch, n_kept=3, decoded=4, kept=[0, 2, 3])
    eng = MuseTalkEngine(models_dir=tmp_path / "m")

    state = eng._load_avatar(_StubRuntime(), pack)

    assert state.cycle_len == 3
    assert [int(f[0, 0, 0]) for f in state.frames] == [1, 3, 4]  # decoded[kept]
    assert len(state.masks) == 3


def test_load_video_pack_rejects_inconsistent_sidecars(tmp_path, monkeypatch):
    # coords has 3 rows but kept says 2 -> refuse rather than misalign
    pack = _video_pack(tmp_path, monkeypatch, n_kept=2, decoded=4, kept=[0, 1])
    import json

    (tmp_path / "musetalk" / "coords.json").write_text(json.dumps([{"bbox": [0, 0, 8, 8], "mask_crop_box": [0, 0, 8, 8]}] * 3))
    eng = MuseTalkEngine(models_dir=tmp_path / "m")

    with pytest.raises(EngineUnavailable, match="inconsistent"):
        eng._load_avatar(_StubRuntime(), pack)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_musetalk_engine.py -k video_pack -v`
Expected: FAIL — `KeyError: 'mask'`（新布局没有 `mask` 键，旧加载器找不到）

- [ ] **Step 3: 实现新加载路径**

In `server/engine/musetalk_engine.py`, replace `_load_avatar` with the dispatcher and add the video loader:

```python
    def _load_avatar(self, rt: _Runtime, pack: AvatarPack) -> _AvatarState:
        prep = pack.prep or {}
        if not prep.get("latents") or not (prep.get("coords") or prep.get("mask")):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} has no musetalk pack; prepare it with "
                "DH_AVATAR_PIPELINE=musetalk"
            )
        root = Path(pack.root)
        if prep.get("coords"):
            return self._load_video_avatar(rt, pack, root, prep)
        return self._load_legacy_avatar(rt, root, prep)

    def _load_legacy_avatar(self, rt: _Runtime, root: Path, prep: dict) -> _AvatarState:
        """A single-frame pack written before video support: N=1."""
        from PIL import Image

        base_rgb = np.array(
            Image.open(root / prep.get("base_image", "normalized.png")).convert("RGB")
        )
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        return _AvatarState(
            frames=base_rgb[None, ...],
            latents=latents,
            bboxes=[[int(v) for v in prep["bbox"]]],
            masks=[np.array(Image.open(root / prep["mask"]).convert("L"))],
            mask_boxes=[[int(v) for v in prep["mask_crop_box"]]],
        )

    def _load_video_avatar(self, rt: _Runtime, pack: AvatarPack, root: Path, prep: dict) -> _AvatarState:
        import json

        from PIL import Image

        from server.avatar.video import decode_frames

        settings = get_settings()
        meta = json.loads((root / prep["dir"] / "meta.json").read_text())
        kept = [int(i) for i in meta["kept"]]
        coords = json.loads((root / prep["coords"]).read_text())
        latents = _load_latents(root / prep["latents"]).to(rt.device, dtype=rt.weight_dtype)
        masks = [
            np.array(Image.open(root / prep["masks"] / f"{i:08d}.png").convert("L"))
            for i in range(len(kept))
        ]
        decoded, _ = decode_frames(
            root / prep["source"],
            target_fps=self.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if not (
            len(kept) == len(coords) == int(latents.shape[0]) == len(masks) == int(meta["frame_count"])
        ) or not kept or max(kept) >= len(decoded):
            raise EngineUnavailable(
                f"avatar {pack.avatar_id} video pack is inconsistent "
                f"(kept={len(kept)}, coords={len(coords)}, latents={int(latents.shape[0])}, "
                f"decoded={len(decoded)})"
            )
        return _AvatarState(
            frames=np.ascontiguousarray(decoded[kept]),
            latents=latents,
            bboxes=[[int(v) for v in c["bbox"]] for c in coords],
            masks=masks,
            mask_boxes=[[int(v) for v in c["mask_crop_box"]] for c in coords],
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_musetalk_engine.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add server/engine/musetalk_engine.py tests/test_musetalk_engine.py
git commit -m "feat(engine): load N-frame video packs, keeping the legacy single-frame path"
```

---

### Task 8: 后台控制台上传视频

**Files:**
- Modify: `admin-ui/src/pages/Avatars.tsx`
- Test: `tests/test_admin_ui.py`（源码守卫）

**Interfaces:**
- Consumes: 现有 `POST /api/admin/avatars`（Task 2/5 后已接受视频）
- Produces: 上传控件 `accept="image/*,video/*"`；选中视频时用 `<video>` 预检时长，超限即时提示

- [ ] **Step 1: 写失败测试**

Append to `tests/test_admin_ui.py`:

```python
def test_avatar_upload_accepts_video():
    # The upload control must offer videos as well as images; the server enforces
    # the real limits, but the widget has to let a video be chosen at all.
    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui"
        / "src"
        / "pages"
        / "Avatars.tsx"
    ).read_text(encoding="utf-8")
    assert 'accept="image/*,video/*"' in source
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_admin_ui.py::test_avatar_upload_accepts_video -v`
Expected: FAIL — assertion（源码仍是 `accept="image/*"`）

- [ ] **Step 3: 实现前端改动**

In `admin-ui/src/pages/Avatars.tsx`:

Replace the `onCreate` guard and add a duration pre-check helper above the component's return:

```tsx
  async function onCreate({ name }: { name: string }) {
    const file = fileList[0]?.originFileObj;
    if (!file) {
      message.warning("请先选择图片或视频");
      return;
    }
    if (file.type.startsWith("video/")) {
      try {
        const seconds = await videoSeconds(file);
        if (seconds > 15) {
          message.error(`视频 ${seconds.toFixed(1)} 秒，超过 15 秒上限，请先剪短`);
          return;
        }
      } catch {
        // Duration is only a convenience; the server is authoritative.
      }
    }
    setCreating(true);
```

Add the helper (module scope, above `export default function Avatars()`):

```tsx
function videoSeconds(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const el = document.createElement("video");
    el.preload = "metadata";
    el.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(el.duration);
    };
    el.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("cannot read video"));
    };
    el.src = url;
  });
}
```

Replace the upload `Form.Item`:

```tsx
          <Form.Item label="驱动图片或视频" required>
            <Upload.Dragger
              accept="image/*,video/*"
              maxCount={1}
              listType="picture"
              beforeUpload={() => false}
              fileList={fileList}
              onChange={({ fileList: next }) => setFileList(next)}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽图片 / 视频到此处</p>
              <p className="ant-upload-hint">支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身</p>
            </Upload.Dragger>
          </Form.Item>
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_admin_ui.py -v` 和 `cd admin-ui && npm run build`
Expected: PASS；构建成功

- [ ] **Step 5: 提交**

```bash
git add admin-ui/src/pages/Avatars.tsx tests/test_admin_ui.py
git commit -m "feat(admin-ui): accept a video upload with a client-side duration pre-check"
```

---

### Task 9: README 文档

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新说明**

In `README.md`:

1. §「组件职责」表中 `AvatarPipeline` 一行，把「上传照片 → 形象包」改为「上传照片/视频 → 形象包」。
2. 配置表（`DH_AVATAR_PIPELINE` 附近）新增三行：

```markdown
| `DH_AVATAR_MAX_VIDEO_SECONDS` | `15.0` | 驱动视频时长上限（秒），超出上传即 422 |
| `DH_AVATAR_MAX_VIDEO_FRAMES` | `375` | 驱动视频解码帧数上限（15s × 25fps） |
| `DH_AVATAR_CACHE_SIZE` | `2` | 引擎按形象缓存的份数；视频形象每份常驻约 1GB 内存 |
```

3. API 表 `POST /api/v1/avatars` 一行，把「multipart `name` + `file` 上传照片」改为「上传照片或短视频（≤15s）」，并注明视频会以动态驱动方式渲染（头部/身体沿用视频运动，嘴部由音频驱动）。
4. 「错误码」段落补 `invalid_video`、`video_too_long`（均 422）。

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: document video-driven avatar creation"
```

---

### Task 10: GPU 端到端测试

**Files:**
- Test: `tests/test_video_avatar_gpu.py`

**Interfaces:**
- Consumes: 真实 MuseTalk 权重、`vendor/MuseTalk/data/video/sun.mp4` 样例、`.venv-musetalk`

- [ ] **Step 1: 写测试**

Create `tests/test_video_avatar_gpu.py`:

```python
"""End-to-end video avatar against the real weights.

Needs the .venv-musetalk interpreter and a CUDA device. The driver video is
built from MuseTalk's own sample clip so it contains a detectable face.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from server.avatar.musetalk_pipeline import MuseTalkAvatarPipeline  # noqa: E402
from server.engine.musetalk_engine import MuseTalkEngine  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.musetalk]

SAMPLE = Path(__file__).resolve().parents[1] / "vendor" / "MuseTalk" / "data" / "video" / "sun.mp4"


def _short_video(dst: Path, frames: int = 40) -> Path:
    cap = cv2.VideoCapture(str(SAMPLE))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    n = 0
    while n < frames:
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
        n += 1
    cap.release()
    out.release()
    return dst


@pytest.mark.skipif(not SAMPLE.exists(), reason="MuseTalk sample video not vendored")
def test_video_avatar_prepares_and_streams(tmp_path):
    # DH_DATA_DIR is deliberately NOT redirected: both the pipeline and the
    # engine resolve their weights from it, and only the output goes to tmp_path.
    src = _short_video(tmp_path / "clip.mp4")
    out = tmp_path / "out"
    cfg = MuseTalkAvatarPipeline().process("v1", src, out, lambda s, p: None)

    assert cfg["musetalk"]["mode"] == "video"
    assert cfg["musetalk"]["frame_count"] > 1
    assert (out / "musetalk" / "coords.json").exists()

    from server.engine.base import AvatarPack

    pack = AvatarPack(
        avatar_id="v1",
        root=out,
        source_image_path=out / "normalized.png",
        config=cfg,
        prep=cfg["musetalk"],
    )
    eng = MuseTalkEngine()
    import asyncio

    asyncio.run(eng.prepare(pack))
    state = eng._avatars["v1"]
    assert state.cycle_len == cfg["musetalk"]["frame_count"]
    if np.array_equal(state.frames[0], state.frames[1]):
        pytest.skip("sample clip's first two frames are identical")

    ctx = eng.new_context(pack)
    frames = asyncio.run(eng.drive(pack, np.zeros(2400, np.float32), 48000, context=ctx))
    assert frames, "video avatar produced no frames"
    assert frames[0].image.shape == (1280, 720, 3)
```

- [ ] **Step 2: 运行（GPU 环境）**

Run: `.venv-musetalk/Scripts/python -m pytest tests/test_video_avatar_gpu.py -m gpu -v`
Expected: PASS（或 `skip`：样例视频缺失 / 前两帧相同）

- [ ] **Step 3: 提交**

```bash
git add tests/test_video_avatar_gpu.py
git commit -m "test(engine): gpu end-to-end for a video-driven avatar"
```

---

## Self-Review

**Spec coverage：**

| Spec 章节 | 任务 |
|---|---|
| §3 决策1 统一帧循环 | Task 3（包）、Task 6（引擎） |
| §3 决策2 乒乓 | Task 6 `cycle_index` |
| §3 决策3 存视频解码加载 | Task 1 `decode_frames`、Task 7 `_load_video_avatar` |
| §3 决策4 ≤15s 上传即 422 | Task 2 `_validate_video`、Task 5 API 用例 |
| §3 决策5 目标帧率重采样 | Task 1 `decode_frames` |
| §4 包格式（含 `kept`/`source`） | Task 3、Task 4 |
| §5 引擎改动 | Task 6、Task 7 |
| §6 解码 + 流水线 | Task 1、Task 4 |
| §7 存储/API/前端 | Task 2、Task 5、Task 8 |
| §8 兼容性（旧包 N=1） | Task 7 `_load_legacy_avatar` |
| §9 配置 | Task 1 Step 3 |
| §10 错误处理 | Task 2（422）、Task 4（MIN_FRAMES）、Task 7（inconsistent） |
| §11 测试策略 | 各任务测试 + Task 10 |
| §12 内存（缓存降到 2） | Task 1 `avatar_cache_size`、Task 6 `self.cache_size` |

**占位符扫描：** 无 TBD/TODO；Task 3 的 `_video_frames` 抛 `NotImplementedError` 是**有意的**中间态，Task 4 立即替换，且 Task 4 的测试在替换前会失败——这是 TDD 序列的一部分，不是遗留占位。

**类型一致性：** `_AvatarState` 字段（`frames/latents/bboxes/masks/mask_boxes` + `cycle_len`）在 Task 6 定义，Task 6/7 的测试与实现一致；`_render(rt, avatar, idxs, prompts)` 在 Task 6 定义并被 `_drive_sync` 与测试桩一致引用；`save_source`（Task 2）在 `_create_avatar_rows` 与测试中同名；`decode_frames(path, target_fps, max_frames, canvas)` 在 Task 1 定义，Task 4/5/7 与测试桩参数名一致；`kept`/`source` 在 Task 3 写入、Task 4 记录、Task 7 读取，字段名一致。

**与现有代码核对（逐个断言对着源文件复核过）：**

- `store.py` 现在抛 `PipelineError`，不是 `AppError`；`save_source_image` 是 `PipelineError` 的唯一使用者，改名后该 import 必须删掉。
- `_create_avatar_rows`（`server/api/avatars.py:67-70`）现在包了一层 `except Exception → AppError("invalid_image", ...)`，必须删，否则 `video_too_long` 会被改写成 `invalid_image`。
- `server/api/admin.py:166` 只是转调 `_create_avatar_rows`，确实无需改动。
- 既有 `meta.json` 的键是 `version/crop/detector/latent_shape/fps/sample_rate`；`latent_shape` 只被 `tests/test_musetalk_pipeline.py:94` 断言，`sample_rate` 无人读——两者都可安全移除。
- `face_crop.png` / `bbox.json` / `landmarks.json` / `mask.png` 只被流水线写、被 `tests/test_musetalk_pipeline.py` 断言，**没有任何 server 端读者**。统一格式后不再产出（spec §4.1 已明确 bbox/landmarks/mask 三项；`face_crop.png` 一并去掉是因为视频会有 N 份）。Task 3 的 `SIDECARS` 与模块 docstring 同步更新。
- `_prep(tmp_path)`（`tests/test_musetalk_engine.py:31`）产出的正是**旧格式**（有 `mask`、无 `coords`），所以它继续当 Task 7 `_load_legacy_avatar` 的夹具，不需要改。
- `me._blend_face_gpu` 被 `tests/test_musetalk_engine_gpu.py:80,90` 以位置参数调用，签名不变。
- `_render` 里对模型是 `rt.unet.model(...)`；Task 6 的桩必须带 `unet` 这一层（已在计划中修正）。
- `_load_video_avatar` 需要 `latents.shape[0]`；`_Moved` 桩没有 `shape`，Task 7 改用会真读文件的 `_Loaded` 桩。
- `AVATAR_CACHE_SIZE` / `base_t` / `mask_t` / `_device_assets` 除引擎自身外无引用，删除安全。
- GPU 测试**不能**改 `DH_DATA_DIR`：流水线与引擎都从它解析权重路径。

**与 spec 的两处偏差（均为实现细节，不改语义）：**

1. `decode_frames` 返回 `np.ndarray [N,H,W,3]` 而非 `list[np.ndarray]`——Task 1 Step 6 同步改 spec §6.1。
2. 统一格式后不再产出 `face_crop.png`（spec §4.1 未提及该文件）。
