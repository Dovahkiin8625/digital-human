# 空闲时播放闭嘴视频 — 设计规格

日期：2026-10-10
分支：`feat/video-driven-avatar`（视频形象能力的延续）

## 1. 背景与目标

视频驱动形象上线后，会话空闲（用户没有说话）时，keepalive 循环推送的是单张
`normalized.png`，画面完全静止；形象详情页的「预览」也是一张静态图。

目标：**空闲时不静止，循环播放闭嘴的源视频画面**，包括：

1. 调试会话中，没有说话的时候；
2. 点开形象详情页预览的时候。

### 用户确认的关键决策

| 决策点 | 选择 |
|---|---|
| 空闲画面内容 | **直接播放源视频原始帧**（不经过模型、零 GPU 开销） |
| 说话开始/结束过渡 | **无缝衔接**：空闲播到哪帧，开口就从同一帧继续；结束后也从当前帧继续 |
| 详情页预览卡片 | **换成自动播放的循环视频**；移除驱动电平滑块与 `preview.png` 端点 |

## 2. 适用范围

- **视频形象**（prepare 时 `mode == "video"`，N = 有效脸帧 ≥ 8）：空闲播放 N 帧
  源画面的乒乓循环。
- **照片形象 / legacy 形象**（N = 1）：乒乓循环退化为静止，行为与今天完全一致，
  仍然显示 `normalized.png`。
- 不重新渲染嘴部：录制源视频时要求人物**闭嘴、不说话**（见 §7 文案）。若源视频
  中嘴部在动，空闲画面会如实显示——这是用户选择的方案前提。

## 3. 总体架构

空闲画面 = 源视频的解码原始帧，即引擎加载视频形象时已构建的
`_AvatarState.frames`（shape `[N,H,W,3]`，RGB），以及磁盘上惰性生成的
`idle.mp4`（仅用于详情页预览）。

```
调试会话：
  keepalive loop ──engine.idle_frame(pack, ctx, pos)──▶ 按相位取源帧 ──▶ sink.push
                     ▲ pos 由 RenderSession 逐 tick 推进
  首块 PCM 到达 ──── 把空闲相位种入 engine context (ctx.phase)
  SPEAKING→IDLE ──── 从 engine context 读回当前相位

详情页：
  GET /api/admin/avatars/{id}/idle.mp4 ──▶ 惰性生成（CPU）──▶ FileResponse
  照片形象回退：GET /api/admin/avatars/{id}/normalized.png
  UI：<video muted autoPlay playsInline loop>
```

## 4. 引擎层设计

### 4.1 帧选择相位（musetalk_engine.py）

`_SessionContext` 增加一个字段：

```python
    # Offset of this context's speech base frames relative to the session's
    # cumulative emitted anchor: the absolute ping-pong cycle position is
    # phase + emitted, and speech frame selection reads
    # cycle_index(phase + emitted + j). Seeding phase = idle position - emitted
    # lets a session whose idle track reached absolute position p start speaking
    # from the identical frame. Whisper/audio bookkeeping is untouched: it is
    # always anchored on emitted, and phase only shifts which base frame is used.
    phase: int = 0
```

`_drive_sync` 中视频组的 cycle 索引计算由

```python
cycle_idxs += [cycle_index(first + j, avatar.cycle_len) for j in range(int(pr.shape[0]))]
```

改为

```python
cycle_idxs += [
    cycle_index(ctx_phase + first + j, avatar.cycle_len)
    for j in range(int(pr.shape[0]))
]
```

其中 `ctx_phase` 从该 session 的 `_SessionContext.phase` 取得（`work` 元组需携带
或在分组循环时从 job 读取）。**注意**：同组只可能是同一 avatar 的多个 session，
但每个 session 的 phase 不同，因此每个 `k` 必须用各自 job 的 ctx.phase，不能共用。

口型提示（whisper prompts）计算路径完全不变：`_prompts_for` 仍以
`first = ctx.emitted` 为锚，phase 不参与任何音频/特征索引。

### 4.2 DriveEngine 接口（engine/base.py）

新增两个方法，默认实现覆盖照片/mock 的退化情形：

```python
    async def idle_frame(self, pack: AvatarPack, context: object | None, pos: int):
        """The source frame to show at cycle position `pos` while idle.

        Returns an HWC RGB numpy array, or None when this pack/engine cannot
        supply source frames (caller falls back to its static canvas).
        Default: unsupported -- keeps the pre-existing static-keepalive path.
        """
        return None

    def idle_position(self, context: object | None) -> int:
        """Current ping-pong cycle position: the frame after the last one used."""
        return 0
```

### 4.3 MuseTalkEngine 实现

```python
    async def idle_frame(self, pack, context, pos):
        avatar = self._avatar_for_idle(pack)   # 见下；失败返回 None
        if avatar is None:
            return None
        return avatar.frames[cycle_index(int(pos), avatar.cycle_len)]

    def idle_position(self, context):
        if isinstance(context, _SessionContext):
            return int(context.phase + context.emitted)
        return 0
```

取 avatar 的辅助逻辑（在 worker 线程上执行——idle_frame 经 keepalive 的 await
链最终落到引擎；实际只是字典查找 + numpy 索引，无 CUDA）：

- 若该 pack 已在 `self._avatars` 缓存：直接用；
- 否则尝试 `self._ensure_avatar(self._ensure_runtime(), pack)`；
- 任何异常（权重缺失、pack 损坏、runtime 不可用）都不允许打断 keepalive：
  捕获后返回 None，RenderSession 回退静态帧。

mock 引擎沿用默认实现（返回 None），现有测试行为不变。

## 5. RenderSession 设计（server/core/render_session.py）

### 5.1 新增状态

```python
        # Ping-pong cycle position of the next idle frame to push. Advanced by
        # one per keepalive tick; seeded INTO the engine context when speech
        # starts and re-read FROM it when speech ends, so idle and speech share
        # one continuous frame position (no visible jump at either boundary).
        self._idle_pos = 0
```

### 5.2 开口瞬间：把空闲相位种入 context

`push_pcm` 中、状态置为 SPEAKING 之前：

```python
            if self.state is not SessionState.SPEAKING:
                # 引擎按 cycle_index(phase + emitted + j) 选帧，而 emitted
                # 跨整段会话累计、空闲时不重置，因此种入的是“空闲绝对位置
                # 减去累计 emitted”的差值；首句两者皆为 0，与直接种位置相同。
                if self._ctx is not None and hasattr(self._ctx, "phase"):
                    self._ctx.phase = int(self._idle_pos) - int(
                        getattr(self._ctx, "emitted", 0)
                    )
```

种入的不是空闲绝对位置本身，而是差值 `_idle_pos − emitted`：选帧时 phase
要叠加会话级**累计**的 emitted（空闲期间也不重置），而空闲时间线是绝对的，
直接种绝对位置会在第二次开口时把之前每段话再计一次；首句两者皆为 0，
与直接种位置完全相同。

`_SessionContext` 类型在运行时由引擎产生，RenderSession 不导入引擎内部类，
统一用鸭子属性判断（mock 引擎的 context 为 None，跳过）。

相位种入只在 LISTENING/IDLE → SPEAKING 的转换点发生一次；持续说话期间
`_idle_pos` 不再使用，不会每块重种。

### 5.3 说话结束：读回当前相位

`maybe_idle` 中 SPEAKING → IDLE 转换时：

```python
        if self.state == SessionState.SPEAKING and now - self.last_audio_at > IDLE_AFTER:
            self.state = SessionState.IDLE
            self._idle_pos = int(self.engine.idle_position(self._ctx))
```

### 5.4 keepalive 循环：逐帧推进

循环体中当前在进入锁之前取静态帧（`frame = self._static_frame()`），在锁内、
SPEAKING re-check 通过后推送。改为：进锁前同时取得引擎源帧与静态回退帧，锁内
优先推源帧（结构与今天一致，re-check 全部保留）：

```python
            source_frame = await self.engine.idle_frame(
                self.pack, self._ctx, int(self._idle_pos)
            )
            frame = None if source_frame is not None else self._static_frame()
            if source_frame is None and frame is None:
                continue
            async with self._lock:
                if self.state in (SessionState.CLOSED, SessionState.ERROR):
                    return
                if self.state is SessionState.SPEAKING:
                    continue
                try:
                    pushed = source_frame if source_frame is not None else frame
                    await self.sink.push(pushed, self._audio_pts)
                except SinkClosed:
                    self._metrics["dropped_frames"] += 1
                    continue
                self._metrics["keepalive_frames"] += 1
                self._audio_pts += period
                if source_frame is not None:
                    self._idle_pos += 1        # 仅源帧占周期位置；静态回退不推进
```

计数与推进的精确规则：

- `idle_frame` 取到帧且 `sink.push` 成功：`keepalive_frames += 1`、
  `_audio_pts += period`、`_idle_pos += 1`；
- 回退静态帧路径：计数/pts 规则同今天，`_idle_pos` **不推进**（静态帧不占周期
  位置，恢复后相位仍然连续）；
- `SinkClosed`：计数 dropped，不推进。

SPEAKING 状态下 keepalive 不推帧（今天的 re-check 逻辑保留），相位也不推进——
说话期间位置由引擎 context 的 emitted 记账，§5.3 在结束时读回。

照片形象（N=1）：`idle_frame` 返回的帧内容恒等于 normalized.png，视觉上与
今天一致。

## 6. 详情页预览设计

### 6.1 新端点：idle.mp4

```
GET /api/admin/avatars/{avatar_id}/idle.mp4
```

- 鉴权与现有 admin 端点一致（X-Admin-Token / 会话）。
- avatar 不存在 → `not_found` 404。
- 视频形象：
  - 文件 `<avatar_dir>/idle.mp4` 已存在 → 直接 `FileResponse`（支持 Range、
    206）；
  - 不存在 → **惰性生成**（见 §6.3），生成成功后返回；
  - 生成失败 → typed error `idle_video_unavailable` / **503**。
- 照片/legacy 形象：返回 `not_found` 404（无 mp4 可播；UI 改取静态图，见下）。

### 6.2 新端点：normalized.png

```
GET /api/admin/avatars/{avatar_id}/normalized.png
```

供照片形象（以及 mp4 不可用时的回退）使用：读取 avatar 目录下
`prep.base_image`（默认 `normalized.png`）并 `FileResponse`。文件缺失 →
`not_found` 404。

### 6.3 idle.mp4 生成（CPU，零 GPU）

新增模块函数（放在 `server/avatar/video.py`，与 `decode_frames` 同处）：

```python
def build_idle_video(avatar_dir: Path, source: Path, settings) -> Path:
    """Render the avatar's ping-pong idle clip from its source video.

    Pure CPU. Decode with the SAME arguments the prepare pipeline used
    (target_fps=settings.fps, max_frames=settings.avatar_max_video_frames,
    canvas=(frame_width, frame_height)), then select the same kept frames the
    engine's idle track plays (see the帧序列 note below).
    """
```

**帧序列**：与引擎空闲帧保持同一组帧。引擎空闲播放的是
`_AvatarState.frames` = 解码序列经 `kept` 筛过的脸帧。为保证预览与会话完全
一致，idle.mp4 必须使用**同一组 kept 帧**：读取 `<dir>/musetalk/meta.json` 的
`kept`，解码后按 `kept` 取帧（与 `_load_video_avatar` 的逻辑一致），再拼
`seq = kept_frames; pingpong = seq + seq[-2:0:-1]`（长度 2N−2，与
`cycle_index` 周期一致）。

- 编码：`cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))`，
  逐帧写入 BGR（解码得到 RGB，写入前翻转）。
- 输出先写到 `idle.mp4.tmp`，写完后 `os.replace` 为 `idle.mp4`，避免并发请求
  读到半截文件；生成过程加一个进程内 `threading.Lock`（模块级 dict，按
  avatar_dir 加锁），第二个并发请求等锁后复用结果。
- meta.json 不存在/无法解析、kept 越界、decoder 返回 0 帧、VideoWriter 打不开：
  抛 `VideoError`，由 API 层翻译为 503。
- 已有形象（本次变更前创建）：首次打开详情页时自动补生成，无需迁移脚本。

### 6.4 前端（AvatarDetail.tsx）

- 移除：`level` / `fetchLevel` 状态、驱动电平 Slider、`fetchPreviewUrl`、
  preview.png 相关 object-URL 逻辑。
- 新增：加载 idle.mp4 的 effect——`fetch("/api/admin/avatars/{id}/idle.mp4",
  {headers: X-Admin-Token})`：
  - 200 → blob → object URL，渲染
    `<video src controls muted autoPlay playsInline loop style={{width:240}} />`；
  - 404（照片形象）→ 改取 `/normalized.png`（同样带 token，blob URL）用
    `<img>` 显示；
  - 503/其他 → 显示错误与「重试」按钮（重试逻辑同今天 preview 的重试）。
  - object URL 在切换/卸载时 revoke（沿用现有 previewRef 模式）。
- 卡片标题保持「预览」；「驱动电平」文案随滑块一并移除。

### 6.5 旧端点移除

- 删除 `GET /api/admin/avatars/{id}/preview.png` 路由（admin.py）及
  `DriveEngine.preview` / `_tone` 中仅供该端点使用的部分：
  - `DriveEngine.preview` 仅被该端点调用 → 删除；
  - `_tone` 仅被 `preview` 使用 → 一并删除；
  - 对应测试 `tests/test_admin_api.py` 中的 preview.png 用例改为 idle.mp4 /
    normalized.png 用例。

## 7. 文案与文档

- 上传区 hint（Avatars.tsx）由
  `支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身`
  改为：
  `支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身；视频中请保持闭嘴、不说话`
- README：
  - 错误码表增加 `idle_video_unavailable`（503）；
  - 视频形象说明段落补充「空闲时循环播放源视频画面，录制时请保持闭嘴」。
- 2026-10-10-video-driven-avatar-design.md 不需要回改（本次新规格为其增量）。

## 8. 测试计划

### 8.1 默认套件（`.venv`，torch-free）

引擎（tests/test_musetalk_engine.py）：

- `_SessionContext.phase` 默认 0；
- 选帧：设置 ctx.phase = p 后，cycle 索引为
  `[cycle_index(p + first + j, n) ...]`；
- 同批两个不同 session、不同 phase：各自帧索引按各自 phase 计算（分组共享
  UNet 但不共享相位）；
- phase 不影响 `_prompts_for`（以 emitted 为锚——用现有 spy 模式断言 whisper
  调用窗口不变）；
- `idle_frame`：已缓存 avatar → 返回对应 cycle 帧；未缓存且 runtime 可用 →
  加载后返回；runtime/权重异常 → 返回 None（不抛）；
- `idle_position` = phase + emitted。

RenderSession（新增/扩展对应测试文件）：

- 空闲：用记录 id​le 调用的假引擎，逐 tick 的 pos 为 0,1,2,...，成功推送后
  推进；
- 开口首块 PCM（LISTENING/IDLE→SPEAKING）：context.phase 被种为差值
  `_idle_pos − ctx.emitted`（首句两者皆为 0，与直接种位置等价；见 §5.2）；
  持续说话的后续块不重种；
- 第二段说话的连续性回归：idle→speech→idle→speech，断言第二段首帧 = 空闲
  下一帧的绝对位置——旧的绝对值种入会把上一段累计 emitted 重复计数；
- IDLE 状态下的 flush 尾块可能让 emitted 超过 `_idle_pos`，使种入差值为负；
  `cycle_index` 用 Python 正取模（−1 % 周期 = 周期−1），负相位恰好对应乒乓
  回摆方向，选帧仍然连续；
- SPEAKING→IDLE（驱动 maybe_idle）：`_idle_pos` 取 engine.idle_position；
- idle_frame 返回 None 时回退静态 normalized.png，且 `_idle_pos` 不推进；
- SinkClosed 时不推进。

API：

- `idle.mp4`：视频形象首次请求触发生成（stub `build_idle_video`）、二次请求
  直接返回文件；Range 请求返回 206；生成器抛错 → 503 `idle_video_unavailable`；
  照片形象 → 404；
- `normalized.png`：照片形象 200、文件缺失 404；
- `preview.png` 路由已移除（请求 → 404）。

idle.mp4 生成：

- 用一个小型合成 mp4（测试夹具级，用 cv2 在 tmp 目录写出 N 帧 + 构造
  meta.json/kept）验证：输出时长 = 2N−2 帧、序列顺序为
  seq + seq[-2:0:-1]、`.tmp` 原子替换；
- kept 越界 / 0 帧 / writer 失败 → VideoError。

前端门禁（tests/test_admin_ui.py，源码断言类）：

- Avatars.tsx hint 含「视频中请保持闭嘴、不说话」；
- AvatarDetail.tsx 含 `idle.mp4`、`<video`，不再含 `preview.png` 与驱动电平
  滑块。

### 8.2 GPU 套件（`.venv-musetalk`，标记 gpu）

在 tests/test_video_avatar_gpu.py 增加一条相位连续性回归：

- prepare → new_context → 模拟空闲推进 p 帧 → 种 phase → drive 一个短块：
  断言首个输出帧的 cycle 位置 = p（与 `idle_frame(p)` 画面一致/连续，可用
  逐帧 cycle_index 序列断言位置相接）；
- drive 结束后 `idle_position(ctx)` = phase + emitted。

### 8.3 构建

- `admin-ui` 下 `npm run build` 保持干净（仅有既存 chunk 体积提示）；
- node:test（reentryGuard）不受影响，继续通过。

## 9. 非目标（YAGNI）

- 不用静音音频驱动 MuseTalk 重新渲染「保证闭嘴」版本（用户已选择原始帧方案）。
- 不做空闲画面的裁剪/选择 UI、不做多种空闲素材。
- 不为 idle.mp4 做一次性批量迁移脚本（惰性生成即可）。
- 不改变说话期间的任何音频/提示计算逻辑。

## 10. 风险与权衡

| 风险 | 评估 |
|---|---|
| 源视频中人物嘴部在动，空闲时可见 | 方案的已知前提；通过上传文案明确「录制时闭嘴」规避 |
| 首次打开旧形象详情页要现场生成 mp4 | 纯 CPU、≤375 帧，秒级；页面有加载态；只发生一次 |
| keepalive 取帧异常影响会话 | 引擎层捕获一切异常并回退静态帧（今天的行为），不允许打断会话 |
| phase 与音频记账耦合引入错位 | phase 只偏移选帧，绝不进入 whisper/特征索引；有专门测试锁定 |
| 详情页与调试空闲画面不一致 | 二者共用同一组 kept 帧与同一个 2N−2 乒乓序列；生成函数与引擎加载逻辑对齐 |
