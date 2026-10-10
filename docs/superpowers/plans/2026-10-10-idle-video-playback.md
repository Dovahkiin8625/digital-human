# 空闲播放闭嘴视频 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 视频形象在会话空闲（无人说话）时循环播放闭嘴的源视频画面而不是静止；形象详情页预览同样改为循环视频；说话开始/结束均无缝衔接。

**Architecture:** 空闲画面 = 引擎已加载的源帧 `_AvatarState.frames`，经同一个 `cycle_index` 乒乓函数取帧；ping-pong 相位存入 `_SessionContext.phase`，由 RenderSession 在开口时种入、在说话结束时读回。详情页预览由新端点提供惰性生成（纯 CPU）的 `idle.mp4`（kept 帧 + 回摆段，共 2N−2 帧）；照片形象回退 `normalized.png`。

**Tech Stack:** Python（FastAPI / numpy / PIL），React + antd + Vite，MuseTalk 1.5（仅 GPU 回归用到）。

**Spec:** `docs/superpowers/specs/2026-10-10-idle-video-playback-design.md` — 计划从规格论证而来，执行者两份都读。

## Global Constraints

- 空闲帧 = **源视频原始帧**，不经过模型、不新增 GPU 开销；照片/legacy 形象（N=1）行为不变。
- 说话开始/结束必须**无缝**：开口首帧 = 空闲当前相位，结束后空闲首帧 = 说话末帧相位。
- phase **只偏移选帧**，绝不进入 whisper/特征/音频索引计算。
- 引擎的加载/构建只能发生在 **musetalk worker 线程**；事件循环线程上的 idle_frame 只做缓存查找，失败回退，绝不允许打断 keepalive。
- 不编辑 `vendor/` 下任何内容；不安装新的 npm/pip 依赖。
- 测试解释器：默认套件 `.venv/Scripts/python.exe -m pytest`（torch/cv2-free）；GPU 用 `.venv-musetalk/Scripts/python.exe -m pytest -m gpu`。禁止裸 `python`。
- 每个任务一个提交，提交信息**不加任何 attribution  trailers**。

---

### Task 1: 引擎支持空闲相位与 idle 取帧接口

**Files:**
- Modify: `server/engine/base.py`（新增 `idle_frame` / `idle_position` 默认实现）
- Modify: `server/engine/musetalk_engine.py`（`_SessionContext.phase`；`_drive_sync` 带 phase 选帧；实现两个新方法）
- Test: `tests/test_musetalk_engine.py`

**Interfaces:**
- Consumes: `_AvatarState.frames`（`[N,H,W,3]` uint8 RGB）、`cycle_index(frame, n)`、`prepare(pack)`（在 worker 上完成缓存）。
- Produces:
  - `DriveEngine.idle_frame(pack, context, pos) -> np.ndarray | None`（async；HWC RGB；None = 调用方回退静态帧）
  - `DriveEngine.idle_position(context) -> int`
  - `_SessionContext.phase: int = 0`

- [ ] **Step 1: 写失败测试**

在 `tests/test_musetalk_engine.py` 末尾追加：

```python
# -- idle phase / idle surface ------------------------------------------

def _video_state(n=3):
    return me._AvatarState(
        frames=np.stack([np.full((1280, 720, 3), 10 * (i + 1), np.uint8) for i in range(n)]),
        latents=[f"latent{i}" for i in range(n)],
        bboxes=[[i, 0, 10, 10] for i in range(n)],
        masks=[np.full((4, 4), 255, np.uint8) for i in range(n)],
        mask_boxes=[[0, i, 4, 4] for i in range(n)],
    )


def test_phase_shifts_frame_selection(engine, tmp_path):
    pack = _pack(tmp_path)
    engine._load_avatar = lambda rt, p: _video_state(3)
    captured: dict = {}

    def render(rt, avatar, idxs, prompts):
        captured["idxs"] = list(idxs)
        return [np.zeros((1280, 720, 3), np.uint8)] * len(idxs)

    engine._render = render

    def run_with_phase(phase):
        ctx = engine.new_context(pack)
        ctx.phase = phase
        job = me._Job(
            pack=pack, ctx=ctx, audio=np.zeros(4800, np.float32),
            sample_rate=48000, future=concurrent.futures.Future(),
        )
        engine._drive_sync([job])

    # One 100 ms block emits 2 frames; n=3 cycle sequence 0,1,2,1,0,...:
    # phase 0 -> [0,1], phase 3 -> [1,0].
    run_with_phase(0)
    assert captured["idxs"] == [0, 1]
    run_with_phase(3)
    assert captured["idxs"] == [1, 0]


def test_phase_does_not_change_prompt_anchors(engine, tmp_path):
    pack = _pack(tmp_path)
    engine._load_avatar = lambda rt, p: _video_state(3)
    anchors: list = []

    def prompts_for(rt, ctx, first, target):
        anchors.append((first, target))
        return _FakePrompts(np.zeros((2, 50, 384), np.float32))

    engine._prompts_for = prompts_for

    def run_with_phase(phase):
        ctx = engine.new_context(pack)
        ctx.phase = phase
        job = me._Job(
            pack=pack, ctx=ctx, audio=np.zeros(4800, np.float32),
            sample_rate=48000, future=concurrent.futures.Future(),
        )
        engine._drive_sync([job])

    run_with_phase(0)
    run_with_phase(3)
    # Whisper bookkeeping is anchored on emitted only; phase must not appear.
    assert anchors == [(0, 2), (0, 2)]


def test_two_sessions_in_one_batch_keep_their_own_phase(engine, tmp_path):
    pack = _pack(tmp_path)
    state = _video_state(3)
    engine._load_avatar = lambda rt, p: state  # same avatar, two sessions
    captured: dict = {}
    engine._render = lambda rt, avatar, idxs, prompts: (
        captured.setdefault("idxs", []).extend(idxs)
        or [np.zeros((1280, 720, 3), np.uint8)] * len(idxs)
    )

    ctx0, ctx1 = engine.new_context(pack), engine.new_context(pack)
    ctx0.phase, ctx1.phase = 3, 0
    jobs = [
        me._Job(pack=pack, ctx=c, audio=np.zeros(4800, np.float32),
                sample_rate=48000, future=concurrent.futures.Future())
        for c in (ctx0, ctx1)
    ]
    engine._drive_sync(jobs)
    # Each session contributes frames at its OWN phase: [1,0] then [0,1].
    assert captured["idxs"] == [1, 0, 0, 1]


@pytest.mark.asyncio
async def test_idle_frame_returns_cached_frames_in_cycle(engine, tmp_path):
    pack = _pack(tmp_path)
    engine._avatars["a"] = _video_state(3)
    frames = [await engine.idle_frame(pack, None, pos) for pos in range(5)]
    assert [int(f[0, 0, 0]) for f in frames] == [10, 20, 30, 20, 10]


@pytest.mark.asyncio
async def test_idle_frame_reloads_evicted_avatar_via_prepare(engine, tmp_path):
    pack = _pack(tmp_path)
    state = _video_state(3)
    prepared = []

    async def prepare(p):
        prepared.append(True)
        engine._avatars["a"] = state

    engine.prepare = prepare
    frame = await engine.idle_frame(pack, None, 1)
    assert prepared == [True]
    assert int(frame[0, 0, 0]) == 20


@pytest.mark.asyncio
async def test_idle_frame_is_none_when_prepare_fails(engine, tmp_path):
    pack = _pack(tmp_path)

    async def boom(p):
        raise RuntimeError("weights gone")

    engine.prepare = boom
    assert await engine.idle_frame(pack, None, 0) is None


def test_idle_position(engine):
    ctx = me._SessionContext()
    assert engine.idle_position(ctx) == 0
    ctx.phase, ctx.emitted = 4, 3
    assert engine.idle_position(ctx) == 7
    assert engine.idle_position(None) == 0


@pytest.mark.asyncio
async def test_default_engine_idle_surface_is_unsupported():
    from server.engine.mock_engine import MockEngine

    eng = MockEngine()
    assert await eng.idle_frame(None, None, 0) is None
    assert eng.idle_position(None) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_musetalk_engine.py -q -k "phase or idle_frame or idle_position or idle_surface"`
Expected: FAIL（`AttributeError: 'MuseTalkEngine' object has no attribute 'idle_frame'` 等）

- [ ] **Step 3: base.py 加默认实现**

在 `server/engine/base.py` 的 `DriveEngine` 中，`preview` 方法**之前**插入：

```python
    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The source frame at ping-pong position `pos` for an idle track.

        Returns an HWC RGB array, or None when this engine/pack cannot supply
        source frames; the caller then falls back to its resting canvas. A
        stateless engine returns None, which keeps the pre-existing static
        keep-alive behaviour.
        """
        return None

    def idle_position(self, context: object | None) -> int:
        """Ping-pong cycle position after the most recently used frame."""
        return 0
```

- [ ] **Step 4: _SessionContext 加 phase**

在 `server/engine/musetalk_engine.py` 的 `_SessionContext` 中，`emitted` 字段之后加：

```python
    # Ping-pong cycle position of the frame just before this context's first
    # speech frame. Speech frames read cycle_index(phase + emitted + j), so a
    # session whose idle track reached position p starts speaking on that same
    # frame. Whisper/audio bookkeeping is untouched: it is anchored on emitted.
    phase: int = 0
```

- [ ] **Step 5: _drive_sync 用各自 job 的 phase 选帧**

在 `_drive_sync` 的分组循环里，把：

```python
                for k in idxs:
                    _, _, first, pr = work[k]
                    cycle_idxs += [cycle_index(first + j, avatar.cycle_len) for j in range(int(pr.shape[0]))]
```

改为：

```python
                for k in idxs:
                    job_i, _, first, pr = work[k]
                    # Every job uses its OWN session's phase: two sessions share
                    # the avatar and the UNet call, never the cycle position.
                    ctx_phase = int(getattr(jobs[job_i].ctx, "phase", 0))
                    cycle_idxs += [
                        cycle_index(ctx_phase + first + j, avatar.cycle_len)
                        for j in range(int(pr.shape[0]))
                    ]
```

- [ ] **Step 6: MuseTalkEngine 实现 idle_frame / idle_position**

在 `MuseTalkEngine` 中 `new_context` 之后插入：

```python
    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The cached avatar's source frame at cycle position `pos`.

        start() already ran prepare() on the worker thread, so the avatar is
        cached; an LRU eviction is repaired by re-running prepare -- also on
        the worker. Never build or load here, on the event-loop thread, where
        it would stall every session. Anything going wrong returns None: a
        keep-alive tick must not break the session, and the caller falls back
        to its static canvas.
        """
        try:
            avatar = self._avatars.get(pack.avatar_id)
            if avatar is None:
                await self.prepare(pack)
                avatar = self._avatars.get(pack.avatar_id)
            if avatar is None:
                return None
            return avatar.frames[cycle_index(int(pos), avatar.cycle_len)]
        except Exception:  # noqa: BLE001 -- keep-alive survives the engine
            return None

    def idle_position(self, context: object | None) -> int:
        if isinstance(context, _SessionContext):
            return int(context.phase + context.emitted)
        return 0
```

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_musetalk_engine.py -q`
Expected: PASS（之前 44 passed，现多出 8 个 → 52 passed, 1 deselected）

- [ ] **Step 8: 提交**

```bash
git add server/engine/base.py server/engine/musetalk_engine.py tests/test_musetalk_engine.py
git commit -m "feat(engine): cycle phase and idle-frame surface"
```

---

### Task 2: RenderSession 空闲播放源帧、说话起止无缝

**Files:**
- Modify: `server/core/render_session.py`（`_idle_pos`；`push_pcm` 种 phase；`maybe_idle` 读回；`_keepalive_loop` 源帧优先、静态回退）
- Test: `tests/test_render_session.py`

**Interfaces:**
- Consumes: `engine.idle_frame(pack, ctx, pos)`（async）、`engine.idle_position(ctx)`、`ctx.phase`。
- Produces: RenderSession 内部 `_idle_pos`（测试会直接观察）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_render_session.py` 末尾追加：

```python
# -- idle source-frame playback -----------------------------------------

class _IdleEngine(MockEngine):
    """Serves solid-colour source frames ping-pong, like the real engine."""

    def __init__(self, n=3):
        super().__init__()
        self.n = n
        self.source = np.stack(
            [np.full((1280, 720, 3), 60 + i, np.uint8) for i in range(n)]
        )
        self.idle_calls: list[int] = []

    class _Ctx:
        phase = 0
        emitted = 0

    def new_context(self, pack):
        return self._Ctx()

    async def idle_frame(self, pack, ctx, pos):
        from server.engine.musetalk_engine import cycle_index

        self.idle_calls.append(int(pos))
        return self.source[cycle_index(int(pos), self.n)]

    async def drive(self, pack, audio, sample_rate, *, context=None):
        frames = await super().drive(pack, audio, sample_rate, context=context)
        if context is not None:
            context.emitted += len(frames)
        return frames

    def idle_position(self, ctx):
        return int(getattr(ctx, "emitted", 0))


class _ImageCapture(FrameBufferSink):
    keepalive = True

    def __init__(self):
        super().__init__()
        self.vals: list[int] = []

    async def push(self, image, pts):
        self.vals.append(int(image[0, 0, 0]))


@pytest.mark.asyncio
async def test_idle_track_plays_source_frames_in_cycle(tmp_path):
    engine = _IdleEngine(n=3)
    session, sink = _session(tmp_path, engine=engine, sink=_ImageCapture())
    await session.start()
    await asyncio.sleep(0.3)
    await session.close()
    # At least the first cycle must be source frames in ping-pong order.
    assert len(sink.vals) >= 5
    assert sink.vals[:5] == [60, 61, 62, 61, 60]


@pytest.mark.asyncio
async def test_speech_start_seeds_the_idle_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_LiveSink())
    await session.start()
    await asyncio.sleep(0.2)
    pos = session._idle_pos
    assert pos > 0
    await session.push_pcm(np.zeros(4800, np.int16).tobytes())
    assert session._ctx.phase == pos
    # A second, still-speaking block must not re-seed the phase.
    await session.push_pcm(np.zeros(4800, np.int16).tobytes())
    assert session._ctx.phase == pos
    await session.close()


@pytest.mark.asyncio
async def test_speech_end_reads_back_the_engine_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_LiveSink())
    await session.start()
    # 0.5 s: five 100 ms blocks, mock emits 2 frames each -> emitted 10.
    await session.push_pcm(np.zeros(24000, np.int16).tobytes())
    session.maybe_idle(now=session.last_audio_at + 3.0)
    assert session.state == SessionState.IDLE
    assert session._idle_pos == 10


@pytest.mark.asyncio
async def test_idle_resumes_from_the_speech_end_position(tmp_path):
    engine = _IdleEngine(n=3)
    session, sink = _session(tmp_path, engine=engine, sink=_ImageCapture())
    await session.start()
    await session.push_pcm(np.zeros(24000, np.int16).tobytes())  # 10 frames
    before = len(engine.idle_calls)
    session.maybe_idle(now=session.last_audio_at + 3.0)
    await asyncio.sleep(0.15)
    await session.close()
    # The idle track after speech asks for frames starting at position 10.
    assert engine.idle_calls[before] == 10


@pytest.mark.asyncio
async def test_static_fallback_does_not_advance_the_position(tmp_path):
    # Default MockEngine has no idle frames (idle_frame -> None): the resting
    # canvas is pushed instead and the cycle position must stay put.
    session, _ = _session_with_canvas(tmp_path)
    await session.start()
    await asyncio.sleep(0.2)
    assert session.metrics["keepalive_frames"] > 0
    assert session._idle_pos == 0
    await session.close()


class _ClosedOnPush(_RecordingSink):
    keepalive = True

    async def push(self, image, pts):
        raise SinkClosed()


@pytest.mark.asyncio
async def test_sink_closed_does_not_advance_position(tmp_path):
    # The engine must supply a source frame so the loop reaches sink.push:
    # with the default MockEngine there is no canvas in this pack, so the
    # push (and thus the SinkClosed path) would never be exercised.
    engine = _IdleEngine(n=3)
    session, _ = _session(tmp_path, engine=engine, sink=_ClosedOnPush())
    await session.start()
    await asyncio.sleep(0.15)
    assert session._idle_pos == 0
    assert session.metrics["dropped_frames"] > 0
    await session.close()
```

注：`SinkClosed` 已在 `server/core/stream_sink.py` 定义；测试文件顶部需要把既有
导入改为 `from server.core.stream_sink import FrameBufferSink, NullSink, SinkClosed`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_render_session.py -q -k "source_frames or seeds or reads_back or resumes or fallback_does_not or closed_does_not"`
Expected: FAIL（`AttributeError: _idle_pos` / `idle_frame` 相关）

- [ ] **Step 3: __init__ 加 _idle_pos**

在 `render_session.py` 的 `__init__` 中，`self._idle_frame_read = False` 之后加：

```python
        # Ping-pong position of the next idle frame. Advanced per successful
        # source-frame tick; seeded INTO the engine context when speech starts
        # and re-read FROM it when speech ends, so idle and speech share one
        # continuous position -- no jump at either boundary.
        self._idle_pos = 0
```

- [ ] **Step 4: push_pcm 开口时种入 phase**

把 `push_pcm` 中的：

```python
            self.last_audio_at = time.monotonic()
            self.state = SessionState.SPEAKING
```

改为：

```python
            if self.state is not SessionState.SPEAKING:
                # First audio after idle: hand the idle position to the engine
                # so its first speech frame is the frame the idle track would
                # have shown next. Duck-typed: a mock engine's context is None.
                if self._ctx is not None and hasattr(self._ctx, "phase"):
                    self._ctx.phase = int(self._idle_pos)
            self.last_audio_at = time.monotonic()
            self.state = SessionState.SPEAKING
```

- [ ] **Step 5: maybe_idle 说话结束时读回位置**

把 `maybe_idle`：

```python
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.IDLE
```

改为：

```python
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.IDLE
            self._idle_pos = int(self.engine.idle_position(self._ctx))
```

- [ ] **Step 6: keepalive 循环改为源帧优先**

把 `_keepalive_loop` 循环体中：

```python
            frame = self._static_frame()
            if frame is None:
                continue
            async with self._lock:
                if self.state in (SessionState.CLOSED, SessionState.ERROR):
                    return
                # Re-check under the lock: speech may have started between the
                # pre-check above and here, and a resting canvas pushed between
                # two speech frames is a visible flicker.
                if self.state is SessionState.SPEAKING:
                    continue
                try:
                    await self.sink.push(frame, self._audio_pts)
                except SinkClosed:
                    self._metrics["dropped_frames"] += 1
                    continue
                # Counted apart from frames_out: nothing drove this frame, and
                # folding it in would make the driving pipeline's fps and
                # latency look better the longer a session sat unused.
                self._metrics["keepalive_frames"] += 1
                self._audio_pts += period
```

改为：

```python
            source_frame = await self.engine.idle_frame(
                self.pack, self._ctx, int(self._idle_pos)
            )
            # Static fallback only when the engine supplies no source frame;
            # both are fetched before the lock, below.
            frame = None if source_frame is not None else self._static_frame()
            if source_frame is None and frame is None:
                continue
            async with self._lock:
                if self.state in (SessionState.CLOSED, SessionState.ERROR):
                    return
                # Re-check under the lock: speech may have started between the
                # pre-check above and here, and a frame pushed between two
                # speech frames is a visible flicker.
                if self.state is SessionState.SPEAKING:
                    continue
                try:
                    pushed = source_frame if source_frame is not None else frame
                    await self.sink.push(pushed, self._audio_pts)
                except SinkClosed:
                    self._metrics["dropped_frames"] += 1
                    continue
                # Counted apart from frames_out: nothing drove this frame, and
                # folding it in would make the driving pipeline's fps and
                # latency look better the longer a session sat unused.
                self._metrics["keepalive_frames"] += 1
                self._audio_pts += period
                if source_frame is not None:
                    # Only source frames occupy cycle positions; a static
                    # fallback must not move the phase the next speech seeds.
                    self._idle_pos += 1
```

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_render_session.py -q`
Expected: PASS（既有数量 + 6 个新测试）

- [ ] **Step 8: 全量默认套件**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿

- [ ] **Step 9: 提交**

```bash
git add server/core/render_session.py tests/test_render_session.py
git commit -m "feat(session): play source frames while idle with seamless speech edges"
```

---

### Task 3: idle.mp4 生成器（纯 CPU）

**Files:**
- Modify: `server/avatar/video.py`（新增 `build_idle_video` 与模块级锁字典）
- Test: `tests/test_idle_video.py`（新建）

**Interfaces:**
- Consumes: `decode_frames(path, target_fps, max_frames, canvas)`、`musetalk/meta.json` 的 `kept`、cv2 `VideoWriter`。
- Produces: `build_idle_video(avatar_dir: Path, source: Path, settings) -> Path`，输出 `<avatar_dir>/idle.mp4`；失败抛 `VideoError`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_idle_video.py`：

```python
"""Contract tests for the idle-video generator.

cv2 lives only in the GPU venv, so it is stubbed with a fake capture (solid
source frames) plus a fake VideoWriter that records the encoded frames; a
centre pixel identifies each source frame.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest

from server.avatar import video
from server.avatar.video import VideoError, build_idle_video

CANVAS = (720, 1280)
SETTINGS = types.SimpleNamespace(
    fps=25, avatar_max_video_frames=375, frame_width=720, frame_height=1280
)


class _FakeCap:
    def __init__(self, frames, fps):
        self._frames, self._fps, self._i = frames, fps, 0

    def isOpened(self):
        return True

    def get(self, prop):
        return self._fps if prop == 5 else len(self._frames)  # CAP_PROP_FPS=5

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


class _Writer:
    instances: list["_Writer"] = []

    def __init__(self, path, fourcc, fps, size, opened=True):
        self.path, self.fps, self.size = path, fps, size
        self.frames: list[np.ndarray] = []
        self.opened = opened
        _Writer.instances.append(self)

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(np.array(frame))

    def release(self):
        # The real writer materialises the file; emulate so the exists() reuse
        # check in build_idle_video works.
        Path(self.path).write_bytes(b"FAKEMP4")


class _FakeCV2:
    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, path):
        return _FakeCap(self.frames, 25.0)

    def VideoWriter(self, path, fourcc, fps, size):
        return _Writer(path, fourcc, fps, size)

    def VideoWriter_fourcc(self, *a):
        return "mp4v"


def _solid(n):
    return [np.full((CANVAS[1], CANVAS[0], 3), i, np.uint8) for i in range(n)]


def _avatar_dir(tmp_path, kept):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    (d / "musetalk" / "meta.json").write_text(json.dumps({"kept": kept}))
    return d


@pytest.fixture(autouse=True)
def _reset_writers():
    _Writer.instances = []
    yield


def _install(monkeypatch, frames):
    monkeypatch.setattr(video, "_cv2", lambda: _FakeCV2(frames))


def test_idle_video_writes_kept_frames_ping_pong(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2, 4, 6])
    _install(monkeypatch, _solid(10))
    out = build_idle_video(d, d / "source.mp4", SETTINGS)
    assert out == d / "idle.mp4"
    writer = _Writer.instances[0]
    assert writer.size == CANVAS
    # seq (4 frames) + seq[-2:0:-1] (2 frames) = 6 frames, 2N-2 period.
    centre = [int(f[640, 360, 0]) for f in writer.frames]
    assert centre == [0, 2, 4, 6, 4, 2]
    assert not (d / "idle.mp4.tmp").exists()


def test_idle_video_flips_rgb_to_bgr_for_the_writer(monkeypatch, tmp_path):
    # One coloured frame: R=10 G=20 B=30 in the decoded RGB must arrive at the
    # writer as BGR (30,20,10).
    frames = [np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)]
    frames[0][:, :] = (10, 20, 30)
    d = _avatar_dir(tmp_path, [0])
    _install(monkeypatch, frames)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert tuple(_Writer.instances[0].frames[0][640, 360]) == (30, 20, 10)


def test_idle_video_is_built_once_then_reused(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 2])
    _install(monkeypatch, _solid(5))
    build_idle_video(d, d / "source.mp4", SETTINGS)
    build_idle_video(d, d / "source.mp4", SETTINGS)
    assert len(_Writer.instances) == 1


def test_missing_meta_is_video_error(monkeypatch, tmp_path):
    d = tmp_path / "av"
    (d / "musetalk").mkdir(parents=True)
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)


def test_kept_outside_decoded_range_is_video_error(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [99])
    _install(monkeypatch, _solid(5))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)


def test_unopenable_writer_is_video_error(monkeypatch, tmp_path):
    d = _avatar_dir(tmp_path, [0, 1])
    _install(monkeypatch, _solid(5))

    class _BrokenCV2(_FakeCV2):
        def VideoWriter(self, path, fourcc, fps, size):
            return _Writer(path, fourcc, fps, size, opened=False)

    monkeypatch.setattr(video, "_cv2", lambda: _BrokenCV2(_solid(5)))
    with pytest.raises(VideoError):
        build_idle_video(d, d / "source.mp4", SETTINGS)
    assert not (d / "idle.mp4.tmp").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_idle_video.py -q`
Expected: FAIL（`ImportError: cannot import name 'build_idle_video'`）

- [ ] **Step 3: video.py 实现生成器**

在 `server/avatar/video.py` 顶部增加 `import threading`，并在文件末尾追加：

```python
#: Per-avatar build locks, so two first-requests for the same idle clip
#: serialize: the loser waits and reuses the file instead of reading a
#: half-written one or building a second copy.
_idle_locks: dict[str, threading.Lock] = {}


def build_idle_video(avatar_dir: Path, source: Path, settings) -> Path:
    """Render ``<avatar_dir>/idle.mp4``: the kept source frames ping-ponged.

    Pure CPU and torch-free. The frame set is exactly what the engine's idle
    track plays -- decode with the prepare pipeline's arguments and select the
    frames named by ``musetalk/meta.json``'s ``kept`` -- then write
    ``seq + seq[-2:0:-1]``, the same 2N-2 period ``cycle_index`` walks. The
    file is built via ``idle.mp4.tmp`` and atomically replaced so a reader can
    never see a partial clip.
    """
    import json
    import os

    avatar_dir = Path(avatar_dir)
    meta_path = avatar_dir / "musetalk" / "meta.json"
    out = avatar_dir / "idle.mp4"
    lock = _idle_locks.setdefault(str(avatar_dir), threading.Lock())
    with lock:
        if out.exists():
            return out
        try:
            meta = json.loads(meta_path.read_text())
            kept = [int(i) for i in meta["kept"]]
        except (OSError, ValueError, KeyError) as exc:
            raise VideoError(f"cannot read kept frames from {meta_path}") from exc
        frames, _ = decode_frames(
            source,
            target_fps=settings.fps,
            max_frames=settings.avatar_max_video_frames,
            canvas=(settings.frame_width, settings.frame_height),
        )
        if not kept or max(kept) >= len(frames):
            raise VideoError(
                f"kept frames are outside the decoded sequence "
                f"(kept={len(kept)}, decoded={len(frames)})"
            )
        seq = frames[kept]
        clip = np.concatenate((seq, seq[-2:0:-1]), axis=0)
        tmp = out.with_name("idle.mp4.tmp")
        cv2 = _cv2()
        writer = cv2.VideoWriter(
            str(tmp),
            cv2.VideoWriter_fourcc(*"mp4v"),
            settings.fps,
            (settings.frame_width, settings.frame_height),
        )
        if not writer.isOpened():
            writer.release()
            tmp.unlink(missing_ok=True)
            raise VideoError("cannot open idle video writer")
        for rgb in clip:
            writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        writer.release()
        os.replace(tmp, out)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_idle_video.py -q`
Expected: PASS（6 个测试）

- [ ] **Step 5: 提交**

```bash
git add server/avatar/video.py tests/test_idle_video.py
git commit -m "feat(avatar): generate the ping-pong idle clip"
```

---

### Task 4: 管理端 idle.mp4 / normalized.png 端点，移除 preview.png

**Files:**
- Modify: `server/api/admin.py`（删 preview.png 路由；加两个新路由；FileResponse 导入）
- Modify: `server/engine/base.py`（删 `preview` 方法与 `_tone`）
- Modify: `tests/test_admin_api.py`（替换 preview 用例）
- Modify: `tests/test_mock_engine.py`（删 preview-empty 用例）

**Interfaces:**
- Consumes: `build_idle_video(avatar_dir, source, settings)`、`store.load_pack(avatar)` 的 `prep.mode/source/base_image`。
- Produces:
  - `GET /api/admin/avatars/{id}/idle.mp4`（视频形象；惰性生成；200/206；照片形象 404；生成失败 503 `idle_video_unavailable`）
  - `GET /api/admin/avatars/{id}/normalized.png`（200/404）

- [ ] **Step 1: 写失败测试**

在 `tests/test_admin_api.py` 中，把 `test_avatar_config_and_preview` 整个函数替换为以下内容：

```python
def test_avatar_config(client):
    client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t"})
    api_key = client.post("/api/admin/tenants", headers=ADMIN, json={"name": "t2"}).json()["api_key"]
    avatar_id = client.post(
        "/api/v1/avatars", headers={"X-API-Key": api_key}, data={"name": "a"},
        files={"file": ("a.png", _png(), "image/png")},
    ).json()["avatar_id"]

    resp = client.patch(
        f"/api/admin/avatars/{avatar_id}/config", headers=ADMIN, json={"mouth_scale": 1.5}
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["mouth_scale"] == 1.5


def _make_video_mode(client, avatar_id):
    """Rewrite the uploaded avatar's pack.json into a video-mode pack."""
    from server.config import get_settings

    root = Path(get_settings().data_dir) / "avatars" / avatar_id
    (root / "pack.json").write_text(
        json.dumps(
            {"musetalk": {"mode": "video", "source": "source.mp4", "base_image": "normalized.png"}}
        )
    )
    return root


def test_normalized_png_is_served(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/normalized.png", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 50


def test_idle_mp4_is_generated_then_served_with_range(client, monkeypatch):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    root = _make_video_mode(client, avatar_id)

    def fake_build(avatar_dir, source, settings):
        assert source == avatar_dir / "source.mp4"
        out = avatar_dir / "idle.mp4"
        out.write_bytes(b"FAKEMP4-DATA")
        return out

    monkeypatch.setattr("server.avatar.video.build_idle_video", fake_build)

    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.content == b"FAKEMP4-DATA"

    # A ranged read is served by FileResponse (206 Partial Content).
    rng = client.get(
        f"/api/admin/avatars/{avatar_id}/idle.mp4", headers={**ADMIN, "Range": "bytes=0-3"}
    )
    assert rng.status_code == 206
    assert rng.content == b"FAKE"


def test_idle_mp4_build_failure_is_503(client, monkeypatch):
    from server.avatar.video import VideoError

    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    _make_video_mode(client, avatar_id)

    def boom(avatar_dir, source, settings):
        raise VideoError("decoder died")

    monkeypatch.setattr("server.avatar.video.build_idle_video", boom)
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "idle_video_unavailable"


def test_idle_mp4_404_for_photo_avatar(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/idle.mp4", headers=ADMIN)
    assert resp.status_code == 404


def test_preview_png_route_is_removed(client):
    api_key = _new_tenant(client, "t")
    avatar_id = _upload_avatar(client, api_key, "a")
    resp = client.get(f"/api/admin/avatars/{avatar_id}/preview.png", headers=ADMIN)
    assert resp.status_code == 404
```

注意：`tests/test_admin_api.py` 顶部当前只有 `io`、`uuid4`、`pytest`、`PIL`，
必须新增两个导入：`import json` 与 `from pathlib import Path`（`_make_video_mode`
用到）。

在 `tests/test_mock_engine.py` 中删除整个 `test_preview_raises_on_an_empty_frame_list`
函数以及不再使用的 `_EmptyEngine` 类、`EngineUnavailable` 导入（删除后若该导入无其他
使用则移除；保留以文件实际引用为准）。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_api.py -q`
Expected: FAIL（新路由 404；preview.png 仍 200）

- [ ] **Step 3: admin.py 更新导入**

把：

```python
from starlette.responses import Response
```

改为：

```python
from starlette.responses import FileResponse, Response
```

并在 `from server.avatar.pipeline import run_job_in_new_session` 之后加一行：

```python
from server.avatar import video
```

- [ ] **Step 4: 删除 preview.png 路由、添加两个新路由**

把整个 `avatar_preview` 函数（`@admin_router.get("/avatars/{avatar_id}/preview.png")`
起到其 `return Response(...)`）替换为：

```python
@admin_router.get("/avatars/{avatar_id}/idle.mp4")
def avatar_idle_video(avatar_id: str, db=Depends(get_db)):
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    if pack.prep.get("mode") != "video":
        # Photo/legacy avatars have no source cycle; the UI uses normalized.png.
        raise AppError("not_found", f"avatar {avatar_id} has no idle video", 404)
    root = store.avatar_dir(avatar_id)
    path = root / "idle.mp4"
    if not path.exists():
        # Built lazily on this threadpool worker: pure CPU, one-time per avatar.
        try:
            path = video.build_idle_video(
                root, root / pack.prep.get("source", "source.mp4"), get_settings()
            )
        except video.VideoError as exc:
            raise AppError("idle_video_unavailable", str(exc), 503) from exc
    return FileResponse(path, media_type="video/mp4")


@admin_router.get("/avatars/{avatar_id}/normalized.png")
def avatar_normalized_png(avatar_id: str, db=Depends(get_db)):
    avatar = db.execute(select(Avatar).where(Avatar.id == avatar_id)).scalar_one_or_none()
    if avatar is None:
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    store = AvatarStore(get_settings().data_dir)
    pack = store.load_pack(avatar)
    path = Path(pack.root) / pack.prep.get("base_image", "normalized.png")
    if not path.exists():
        raise AppError("not_found", "avatar image not found", 404)
    return FileResponse(path, media_type="image/png")
```

- [ ] **Step 5: base.py 删除 preview 与 _tone**

删除 `DriveEngine.preview` 整个方法，以及模块级 `_tone` 函数。删除后确认
`EngineUnavailable` 在 base.py 仍被其他代码使用（`create_engine` 等——保留该类）。

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_api.py tests/test_mock_engine.py -q`
Expected: PASS

- [ ] **Step 7: 全量默认套件**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿

- [ ] **Step 8: 提交**

```bash
git add server/api/admin.py server/engine/base.py tests/test_admin_api.py tests/test_mock_engine.py
git commit -m "feat(admin): idle clip and resting image endpoints"
```

---

### Task 5: 详情页改循环视频、上传提示闭嘴要求、README

**Files:**
- Modify: `admin-ui/src/pages/AvatarDetail.tsx`（循环视频/图片回退；移除滑块 preview 逻辑）
- Modify: `admin-ui/src/pages/Avatars.tsx`（上传 hint 文案）
- Modify: `tests/test_admin_ui.py`（源码门禁更新）
- Modify: `README.md`（端点表、错误码、闭嘴录制说明）

**Interfaces:**
- Consumes: `GET /api/admin/avatars/{id}/idle.mp4`、`GET .../normalized.png`。
- Produces: 前端预览 `<video muted autoPlay playsInline loop>`；照片形象 `<img>`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_admin_ui.py` 中：

把 `test_avatar_upload_accepts_video` 保持不变。追加：

```python
def test_avatar_upload_hint_asks_for_closed_mouth():
    assert "视频中请保持闭嘴、不说话" in _avatars_source()


def _detail_source():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "admin-ui" / "src" / "pages" / "AvatarDetail.tsx"
    ).read_text(encoding="utf-8")


def test_detail_preview_is_a_loop_video_with_image_fallback():
    source = _detail_source()
    assert "idle.mp4" in source
    assert "<video" in source or "video" in source
    assert "muted" in source and "autoPlay" in source and "loop" in source
    assert "normalized.png" in source
    assert "preview.png" not in source


def test_detail_has_no_drive_level_slider():
    source = _detail_source()
    assert "驱动电平" not in source
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_ui.py -q`
Expected: FAIL

- [ ] **Step 3: 重写 AvatarDetail.tsx**

用以下内容**整体替换** `admin-ui/src/pages/AvatarDetail.tsx`：

```tsx
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Row,
  Space,
  Spin,
  Typography,
} from "antd";
import { ArrowLeftOutlined, ReloadOutlined } from "@ant-design/icons";
import { api, getToken } from "../api";
import StatusTag from "../components/StatusTag";
import type { AvatarInfo } from "../types";

type PreviewKind = "video" | "image";

async function fetchObjectUrl(
  path: string
): Promise<{ url: string }> {
  // A plain <video>/<img> cannot attach X-Admin-Token; fetch with the header
  // and hand the element an object URL of the blob.
  const resp = await fetch(path, {
    headers: { "X-Admin-Token": getToken() ?? "" },
  });
  if (!resp.ok) {
    const err = new Error(`request failed: ${resp.status}`) as Error & {
      status: number;
    };
    err.status = resp.status;
    throw err;
  }
  return { url: URL.createObjectURL(await resp.blob()) };
}

export default function AvatarDetail() {
  const { id } = useParams();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [mouth, setMouth] = useState(1);
  const [expression, setExpression] = useState(1);
  const [preview, setPreview] = useState("");
  const [previewKind, setPreviewKind] = useState<PreviewKind>("video");
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const previewRef = useRef("");
  const { message } = AntApp.useApp();

  async function refresh() {
    if (!id) return;
    const data = await api<AvatarInfo>(`/api/admin/avatars/${id}`);
    setInfo(data);
    setMouth(data.config.mouth_scale ?? 1);
    setExpression(data.config.expression_intensity ?? 1);
  }

  useEffect(() => {
    refresh()
      .catch((err) => setError((err as Error).message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function loadPreview() {
    if (!id) return;
    setPreviewLoading(true);
    setPreviewError("");
    let url = "";
    let kind: PreviewKind = "video";
    try {
      try {
        ({ url } = await fetchObjectUrl(
          `/api/admin/avatars/${id}/idle.mp4?t=${Date.now()}`
        ));
      } catch (err) {
        // 404 = no idle clip (photo avatar): fall back to the resting image.
        // Any other status is a real failure shown below with a retry.
        if ((err as { status?: number }).status !== 404) {
          throw err;
        }
        ({ url } = await fetchObjectUrl(
          `/api/admin/avatars/${id}/normalized.png?t=${Date.now()}`
        ));
        kind = "image";
      }
      // Replace and revoke the previous object URL so blobs do not leak.
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      previewRef.current = url;
      setPreviewKind(kind);
      setPreview(url);
    } catch (err) {
      setPreviewError((err as Error).message);
    } finally {
      setPreviewLoading(false);
    }
  }

  useEffect(() => {
    void loadPreview();
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function save() {
    if (!id) return;
    setSaving(true);
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/config`, {
        method: "PATCH",
        body: { mouth_scale: mouth, expression_intensity: expression },
      });
      message.success("参数已保存");
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function retry() {
    if (!id) return;
    try {
      setError("");
      await api(`/api/admin/avatars/${id}/retry`, {
        method: "POST",
        body: {},
      });
      await refresh();
    } catch (err) {
      message.error((err as Error).message);
    }
  }

  if (loading) {
    return (
      <div style={{ display: "grid", placeItems: "center", padding: 64 }}>
        <Spin tip="加载中…">
          <div style={{ width: 120, height: 60 }} />
        </Spin>
      </div>
    );
  }

  // The fetch failed or the avatar is gone — show why instead of spinning forever.
  if (!info) {
    return (
      <Space direction="vertical" size={16}>
        <Alert
          type="error"
          showIcon
          message="无法加载该形象"
          description={error || "形象不存在或已被删除"}
        />
        <Link to="/avatars">
          <Button icon={<ArrowLeftOutlined />}>返回列表</Button>
        </Link>
      </Space>
    );
  }

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Space align="center" wrap>
        <Link to="/avatars">
          <Button icon={<ArrowLeftOutlined />} type="text">
            返回
          </Button>
        </Link>
        <Typography.Title level={3} style={{ margin: 0 }}>
          {info.name}
        </Typography.Title>
        <StatusTag status={info.status} />
      </Space>

      <Descriptions
        column={{ xs: 1, sm: 2, md: 3 }}
        bordered
        size="small"
        items={[
          { key: "id", label: "形象 ID", children: info.avatar_id },
          {
            key: "created",
            label: "创建时间",
            children: new Date(info.created_at).toLocaleString(),
          },
          {
            key: "config",
            label: "当前参数",
            children: `口型 ${(info.config.mouth_scale ?? 1).toFixed(2)} · 表情 ${(
              info.config.expression_intensity ?? 1
            ).toFixed(2)}`,
          },
        ]}
      />

      {info.error && (
        <Alert
          type="error"
          showIcon
          message="生成失败"
          description={info.error}
          action={
            <Button size="small" danger onClick={retry}>
              重试
            </Button>
          }
        />
      )}
      {error && <Alert type="error" showIcon message={error} />}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title="生成参数">
            <Typography.Text>口型幅度 {mouth.toFixed(2)}</Typography.Text>
            <Slider min={0} max={2} step={0.05} value={mouth} onChange={setMouth} />
            <Typography.Text>表情强度 {expression.toFixed(2)}</Typography.Text>
            <Slider min={0} max={2} step={0.05} value={expression} onChange={setExpression} />
            <Button type="primary" onClick={save} loading={saving}>
              保存参数
            </Button>
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card title="预览">
            {previewError ? (
              <Empty description={`预览不可用（${previewError}）`}>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => {
                    void loadPreview();
                  }}
                >
                  重试
                </Button>
              </Empty>
            ) : (
              <Spin spinning={previewLoading}>
                <div style={{ minHeight: 120 }}>
                  {preview && previewKind === "video" ? (
                    <video
                      src={preview}
                      controls
                      muted
                      autoPlay
                      playsInline
                      loop
                      style={{ width: 240 }}
                    />
                  ) : preview ? (
                    <img src={preview} alt="preview" width={240} />
                  ) : null}
                </div>
              </Spin>
            )}
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
```

注意：上面的「生成参数」卡片仍使用 antd 的 `Slider`，因此 antd 导入列表中必须
**保留 `Slider`**；`Image`（antd 的）不再使用，不要导入。请在写入后核对导入块与
文件主体一致（即导入含 `Slider`、不含 `Image`）。

- [ ] **Step 4: Avatars.tsx 更新 hint 文案**

把 `admin-ui/src/pages/Avatars.tsx` 中：

```tsx
              <p className="ant-upload-hint">支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身</p>
```

改为：

```tsx
              <p className="ant-upload-hint">支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身；视频中请保持闭嘴、不说话</p>
```

- [ ] **Step 5: README 更新**

5a. 端点表中把：

```markdown
| GET | `/api/admin/avatars/{avatar_id}/preview.png` | 静态预览（可选 `?level=0..1` 驱动口型） |
```

替换为两行：

```markdown
| GET | `/api/admin/avatars/{avatar_id}/idle.mp4` | 空闲循环视频（视频形象；首次请求时生成，支持 Range） |
| GET | `/api/admin/avatars/{avatar_id}/normalized.png` | 静态画面（照片形象与回退） |
```

5b. 把第 132 行租户上传端点行末的说明：

```markdown
视频形象以动态驱动方式渲染：头部/身体沿用视频运动，嘴部由音频驱动
```

改为：

```markdown
视频形象以动态驱动方式渲染：头部/身体沿用视频运动，嘴部由音频驱动；空闲时循环播放源视频画面，录制时请保持闭嘴、不说话
```

5c. 把错误码句：

```markdown
上传环节错误码：`invalid_video`（视频无法解码，422）、`video_too_long`（时长或帧数超限，422）、`file_too_large`（413）。
```

改为：

```markdown
上传环节错误码：`invalid_video`（视频无法解码，422）、`video_too_long`（时长或帧数超限，422）、`file_too_large`（413）、`idle_video_unavailable`（空闲视频生成失败，503）。
```

- [ ] **Step 6: 跑 Python 门禁**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_ui.py -q`
Expected: PASS

- [ ] **Step 7: 前端构建**

Run（在 `admin-ui/` 下）：`npm run build`
Expected: `tsc` 无错误、vite 构建成功；仅有既存的 chunk 体积提示。

- [ ] **Step 8: 全量默认套件**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿

- [ ] **Step 9: 提交**

```bash
git add admin-ui/src/pages/AvatarDetail.tsx admin-ui/src/pages/Avatars.tsx tests/test_admin_ui.py README.md
git commit -m "feat(admin-ui): looped idle preview with closed-mouth guidance"
```

---

### Task 6: GPU 相位连续性回归

**Files:**
- Modify: `tests/test_video_avatar_gpu.py`

**Interfaces:**
- Consumes: 真实 prepare/engine；`ctx.phase`、`engine.idle_position(ctx)`、`engine.idle_frame(pack, ctx, pos)`。

- [ ] **Step 1: 加回归测试**

在 `tests/test_video_avatar_gpu.py` 末尾追加：

```python
@pytest.mark.skipif(not SAMPLE.exists(), reason="MuseTalk sample video not vendored")
def test_idle_phase_carries_seamlessly_into_speech(tmp_path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    src = _short_video(out / "clip.mp4")
    cfg = MuseTalkAvatarPipeline().process("v3", src, out, lambda s, p: None)

    pack = AvatarPack(
        avatar_id="v3",
        root=out,
        source_image_path=out / "normalized.png",
        config=cfg,
        prep=cfg["musetalk"],
    )
    eng = MuseTalkEngine()
    import asyncio

    asyncio.run(eng.prepare(pack))
    n = int(cfg["musetalk"]["frame_count"])

    # Simulate an idle track that reached position p (= the position the next
    # idle frame would use), as RenderSession does.
    p = 3
    idle_here = asyncio.run(eng.idle_frame(pack, None, p))
    ctx = eng.new_context(pack)
    ctx.phase = p
    frames = asyncio.run(eng.drive(pack, np.zeros(4800, np.float32), 48000, context=ctx))
    assert len(frames) >= 2
    # The first speech frame uses the same base frame the idle track would have
    # shown next: cycle position p. (MuseTalk edits only the mouth region.)
    from server.engine.musetalk_engine import cycle_index

    assert cycle_index(p, n) == cycle_index(ctx.phase + 0, n)
    # Outside the edited mouth box, idle-at-p and first speech frame agree.
    bbox = [int(v) for v in cfg["musetalk"]["bbox"]]
    x1, y1, x2, y2 = bbox
    rest = np.delete(np.delete(idle_here, slice(y1, y2), axis=0), slice(x1, x2), axis=1)
    got = np.delete(np.delete(frames[0].image, slice(y1, y2), axis=0), slice(x1, x2), axis=1)
    assert np.array_equal(rest, got)
    # The engine reports the position after the emitted speech frames.
    assert eng.idle_position(ctx) == ctx.phase + ctx.emitted
```

- [ ] **Step 2: 跑 GPU 回归**

Run: `.venv-musetalk/Scripts/python.exe -m pytest tests/test_video_avatar_gpu.py -m gpu -q`
Expected: PASS（3 个 GPU 测试，约 30–60 秒）

- [ ] **Step 3: 提交**

```bash
git add tests/test_video_avatar_gpu.py
git commit -m "test(engine): gpu phase continuity from idle into speech"
```

---

## Self-Review

**1. Spec coverage:**

- §4.1 phase 字段 + 带 phase 选帧 → Task 1 Steps 4–5。
- §4.2 base 接口 → Task 1 Step 3。
- §4.3 MuseTalkEngine 实现（含 worker 上补 prepare、异常回 None）→ Task 1 Step 6。
- §5.1 `_idle_pos` → Task 2 Step 3。
- §5.2 开口种 phase（duck-typed、只在转换点）→ Task 2 Step 4。
- §5.3 结束读回 → Task 2 Step 5。
- §5.4 keepalive 源帧优先/静态回退/计数推进规则 → Task 2 Step 6。
- §6.1 idle.mp4 端点全部状态码 → Task 4 Step 4。
- §6.2 normalized.png 端点 → Task 4 Step 4。
- §6.3 生成器（kept、2N−2、原子替换、锁、旧形象惰性）→ Task 3。
- §6.4 前端视频/404 回退/503 重试/URL revoke → Task 5 Step 3。
- §6.5 删 preview.png 路由 / DriveEngine.preview / _tone → Task 4。
- §7 文案（上传 hint、README 错误码与说明）→ Task 5。
- §8.1/8.2/8.3 测试 → Tasks 1–6 各自覆盖。

**2. Placeholder scan:** 无 TBD/TODO；所有代码块完整；签名均定义。

**3. Type consistency:** `idle_frame(pack, context, pos)` 在 base、musetalk、RenderSession 调用点三处一致；`idle_position(context)` 一致；`build_idle_video(avatar_dir, source, settings)` 定义、调用、测试 stub 三处一致；`prep` 键（mode/source/base_image）与 store/musetalk pipeline 实际写出一致。
