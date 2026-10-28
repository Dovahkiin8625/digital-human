# Live2D 数字人 — 设计文档

日期：2026-10-22
状态：已与需求方对齐（并存不替换 / 桌面瘦客户端 / v1 仅文字输入 / 收进 dh-admin-ui，四项均经确认）
关联：`docs/superpowers/specs/2026-10-08-digital-human-service-design.md`、`2026-10-10-video-driven-avatar-design.md`

## 1. 目标与范围

在**同一个服务**内增加 Live2D 数字人形态，与现有的服务端驱动数字人（MuseTalk；后来又加入了纯 CPU 的 Wav2Lip / `wav2lip`）**并存**。

Live2D 是后台管理台的一等功能，**三个页面都能选 Live2D 形象**：

- **形象管理** —— 上传、校验、列出、删除 Live2D 模型。
- **会话调试** —— 选中 Live2D 形象时，渲染路径从 WebRTC 视频轨切换为 canvas。
- **数字人会话** —— 同上，并走完整的 LLM → TTS → 口型链路。
- **系统 Tab** —— Live2D 的开关、SDK 路径与构图参数。

以及一个桌面宠物：Electron **独立最小窗口**（透明置顶、鼠标穿透、托盘），复用同一份 Cubism 渲染组件，渲染层本地、服务端远程。

服务端的对话能力（LLM / TTS）完全复用现有实现。

### 1.0 对原设计的修订（2026-10-22，需求方确认）

原设计曾把 Live2D 渲染层做成独立的产品客户端、由 FastAPI 挂在 `/live2d`，且明确「dh-admin-ui 是调试台，不宜混入」。**这两条已被推翻**：

| 原设计 | 修订后 |
|---|---|
| 独立的 Live2D Vite app，挂在 `/live2d` | **收进 `dh-admin-ui`**。Cubism 渲染层做成可复用组件，管理台三个页面直接用。只维护一份前端 |
| Electron 加载同一份独立 bundle | Electron 加载 `dh-admin-ui` 构建产物里的**桌宠路由**（只有 canvas + 输入框，不带管理台 UI） |

代价：`dh-admin-ui` 需要接入 vitest（它此前没有测试运行器），且桌宠与管理台共用一份构建产物。渲染层落在 `dh-admin-ui/src/live2d/`，其测试用 vitest。

### 1.0.1 再次修订（2026-10-27）

上表里「桌宠收进 `dh-admin-ui`」这条后来又被推翻了：桌宠最终**独立成第三个工程**
`dh-pet/`（与 `dh-server/`、`dh-admin-ui/` 平级）。它后来又整个搬出了这个仓库，
成了 `E:/project/dh-pet` 下**自己独立**的应用：自带模型目录、自带 Cubism SDK、
直连大模型与语音合成，不再需要本服务。
彼此之间只有 HTTP API 调用。

| 2026-10-22 的形态 | 2026-10-27 的形态 |
|---|---|
| 桌宠是 `dh-admin-ui` 里的一个 hash 路由，打包时用 `--mode pet` 再打一遍 | 桌宠是独立工程 `dh-pet/`，自带 `package.json`、Electron 外壳与打包 |
| 渲染层只有一份，`dh-admin-ui/src/live2d/`，桌宠与管理台共用 | 渲染层各有一份：`dh-admin-ui/src/live2d/` 与 `dh-pet/src/live2d/` |
| 桌宠的构建产物混在管理台的 `dist/` 里 | 各自 `dist/`，互不干扰；桌宠用相对路径打包（file:// 下才认得 assets） |

代价是渲染层有两份，改动要同步两处 —— 这是「工程独立、不共享源码」换来的。
桌宠的页面底色改为透明（`html` / `body` / `#root`），不再套管理台的主题。

### 1.1 非目标（YAGNI）

- **不做语音输入（ASR）**。v1 仅文字。服务里没有 ASR 代码（`dh-server/dh_server/` 下的 whisper 引用是 MuseTalk 的音频特征提取器，不是识别器），引入它等于新增一个子系统。
- **不替换、不改动现有视频路径**。MuseTalk 的引擎、制作流水线、WebRTC 出口、信令协议一行不改。
- **不给 Live2D 开新的 session 类型**。理由见 §3.2。
- **不做主动搭话 / 服务端推送 / 空闲时的宠物发言**。同理，随 session 一起去。
- **不碰服务端渲染的任何代码路径**。Live2D 不产出视频帧，因此不进入 `DriveEngine`。

### 1.2 与现有设计的关系

原设计 §1 把「服务边界」定为「音频驱动 + 渲染 + 视频流输出」，且「LLM/ASR 由调用方自建」。此后 `dh-server/dh_server/llm/` 与 `dh-server/dh_server/tts/` 已经收回服务内，`dh-admin-ui` 的 Chat 页也已跑通「LLM → 句子队列 → TTS → 驱动」全链路。本设计顺着这个已成事实的边界继续。

## 2. 背景：现状与差距

**Live2D 与现有服务的根本差异**：Live2D 是**客户端渲染**。Cubism SDK 在浏览器 / Electron 里用 WebGL 绘制，服务端一个像素都不产出。服务端从「渲染视频」缩成「在语义粒度上给出时机与情绪」。

这个差异决定了本设计的大部分内容。

**现状差距**（已核实）：

| 现状 | 差距 |
|---|---|
| `Avatar` 表没有「形态 / 模型」列 | 无法区分两种形象形态 |
| LLM / TTS 只挂在 admin 面：`/api/admin/llm/chat/stream`、`/api/admin/tts` | 产品客户端没有业务面的 LLM/TTS 端点可用 |
| 无 Live2D 模型注册与托管 | 需要新增 |
| `dh-admin-ui` 是调试台 | 产品客户端鉴权与生命周期都不同，不宜混入 |
| **`/api/v1/*` 目前完全无鉴权。** README 关于 `X-API-Key` 的描述已过时 —— 鉴权代码在 no-tenant 重构中被删除，`dh-server/dh_server/api/admin.py` 明写「no auth dependencies, all endpoints public」 | 桌面宠物是**第一个连远程服务的客户端**，而模型文件是部署方自己的资产，必须单独鉴权。见 §4.3 |

**已有的有利条件**：

- 迁移是手写的：`_SCHEMA_MIGRATIONS`（`dh-server/dh_server/db/session.py`）是 `(table, column, ddl)` 元组，`init_db()` 每次幂等执行。加字段只需加一行。
- `Avatar.config_json` 是 JSON 列，Live2D 专属字段有现成落点。
- `dh-server/dh_server/llm/openai.py`、`dh-server/dh_server/tts/minimax.py` 已是 provider 无关实现，加端点是路由层工作，不是重写。
- TTS 音频由服务端产出、客户端解码播放，这条链路 `Chat.tsx` 已跑通（MiniMax 上限 44100Hz vs 服务 48000Hz，重采样在浏览器侧完成）。

## 3. 关键决策

### 3.1 在同一个服务内做，不新开服务

现有服务真正的资产不是 MuseTalk，是**对话脑 + 语音管线 + 会话状态机**。新开服务要么重写、要么复制这些 —— 复制的是最贵的部分。

同时注意：现有 `DriveEngine.drive() -> list[DriveFrame]` 返回 numpy 帧数组，**Live2D 不产出帧**。把它塞进这个接口要么写一个把参数序列伪装成假帧的适配器（纯浪费），要么在旁边开一条平行传输通道 —— 后者本质上是本设计，只是名字不同。所以选择**共用大脑、分开渲染路径**：会话与视频通道并列存在，互不干涉。

### 3.2 口型完全在客户端；v1 不开 session

**口型不进服务端。** 客户端已从 TTS 拿到 WAV 的 `AudioBuffer` 并正要送进扬声器，驱动嘴形的就是**同一个音频节点**。因为读的是正在发声的那个源，口型与声音**天然零漂移**。

替代方案（服务端做音素对齐后推参数流）会引入一个与实际播放不同步的时钟，恰好是这个方案要避免的。而且现有链路里音频由服务端合成，服务端推的时长未必等于客户端解码播放的实际时长 —— **代价高，收益为负**。

**因此 v1 不需要 session。** 文字输入下客户端自足：模型文件来自服务端，LLM 与 TTS 是普通 HTTP/SSE 调用，没有任何东西需要服务端主动推。真要会话级指标、主动搭话、断线重连，那是 v2，届时加一条 cue 通道即可，现在不必留疤。

### 3.3 渲染层收进 dh-admin-ui，两个入口共用一份构建

Cubism 渲染层做成 `dh-admin-ui/src/live2d/` 下的可复用模块，由一个 `<Live2DStage>` React 组件封装。管理台的三个页面（形象 / 调试 / 会话）直接使用它，**不再有第二个前端应用**。

Electron 加载 `dh-admin-ui` 的构建产物，指向**桌宠路由** —— 该路由只渲染 canvas + 一个输入框，不含管理台的任何 UI、导航或状态。开发量比另建一个窗口小，而渲染组件仍只有一份。

Electron **不加载远程 URL**，而是加载本地 bundle、服务地址作为配置项。启动瞬时、无白屏闪烁、服务重启期间宠物照常呼吸。

「瘦客户端」的含义是**只有渲染层在本地**：LLM、TTS、鉴权全在远程服务端。Electron 里**不带 Python、不带 torch**。

### 3.4 一个 Live2D 形象不需要新的 session 模型

Live2D 不参与会话，模型字段与真人形象字段（bbox / crop / mask / prep 张量）零重叠。此时另开一张表的收益只有「字段不打架」，代价却是两套列表 / 删除 / 鉴权路径 —— 不划算。故并入 `avatars`，形态由该形象所属的**数字人模型**区分（`avatars.drive_model`）。

### 3.5 管理台里的 Live2D 走 admin 面

管理台三个页面由操作者在同一台机器上操作，走 admin 面。Live2D 的模型文件托管与
LLM / TTS 端点都只有一份，管理台与服务共用：

- 模型文件托管端点（`/api/v1/live2d/avatars/{id}/model/{path}`）没有凭据：
  桌宠独立出去之后，没有第二个需要它的客户端了。
- LLM / TTS 端点只有 admin 面一份（`/api/admin/llm/chat/stream`、`/api/admin/tts`）。

这避免了为了管理台再复制一套 admin 版端点。

## 4. 服务端设计

### 4.1 数据模型：一行迁移

`Avatar` 增加一列，记录**这个形象由哪个数字人模型驱动**，默认 `'musetalk'`。现有数据原地不动：

```python
# dh-server/dh_server/db/session.py 的 _SCHEMA_MIGRATIONS 增加：
("avatars", "drive_model", "ALTER TABLE avatars ADD COLUMN drive_model VARCHAR(32) NOT NULL DEFAULT 'musetalk'"),
```

遵循该元组既有的幂等模式（`has_column` 检查后跳过）。SQLAlchemy 模型上对应加 `drive_model: Mapped[str] = mapped_column(String(32), default="musetalk", nullable=False)`。Live2D 形象记 `drive_model='live2d'`。

这一列是「一个模型 = 一条流水线 + 一个引擎」的落点：Live2D 在
`dh-server/dh_server/model_registry.py` 里注册**只有流水线、没有引擎** —— 客户端渲染，服务端
不产像素。

Live2D 专属字段（模型目录、`model3.json` 入口、默认表情、动作组、emotion→expression 映射）全部放已有的 `config_json`，不新增列：

```json
{
  "model": {
    "entry": "model.model3.json",
    "scale": 0.2,
    "position": [0, 0, 0]
  },
  "expressions": { "neutral": 0, "smile": 3, "think": 5, "sad": 7 },
  "default_motion": "Idle",
  "motion_groups": ["Idle", "TapBody", "PanFace"]
}
```

`scale` / `position` 决定了不同分辨率模型在 canvas 上的初始构图，必须可配 —— 模型自带的原始尺寸与缩放范围差异很大。

### 4.2 模型注册与托管

新包 `dh-server/dh_server/live2d/`：

- **`registry.py`** —— 注册与校验。校验项：`model3.json`（或 `model.model3.json`）存在且可解析、被引用的 `.moc3` 与纹理文件均存在、`expressions` / `motions` 引用的 id 在模型里真实存在。**注册即验证**，不把错误留到客户端渲染时才炸。
- **`archive.py`** —— 上传 zip 的安全解压与入口文件探测（见 2026-10-23 的形象管理设计）。
- 模型文件的托管端点在 `dh-server/dh_server/api/live2d.py`：`GET /api/v1/live2d/avatars/{id}/model/{path}`。

> ⚠️ 模型文件**必须走这个端点下发**，不能裸静态暴露目录。模型是按部署配置的资产，不是公开 CDN 内容。

### 4.3 业务面 LLM 与 TTS

现有 LLM / TTS 只在 admin 面上（`dh-server/dh_server/llm/openai.py`、`dh-server/dh_server/tts/minimax.py` 的 provider 实现被两层端点共用）：

| 端点 | 说明 |
|---|---|
| `POST /api/admin/llm/chat/stream` | `text/event-stream`，每帧 `{"delta": ...}` / `{"done": true}` / `{"error": ...}` |
| `POST /api/admin/tts` | `{"text", "voice_id"}` → `audio/wav`。沿用「服务端只套 WAV 头、重采样放浏览器」的既有做法，服务端不引入音频编解码依赖 |

**没有另开业务面端点**：`/api/v1/chat/stream` 与 `/api/v1/tts` 最终没有实现，
管理台**独占 admin 面的这两个端点**。两处调用共用一套实现与一套路由，比多开一份
业务面端点少一处分叉。

#### 鉴权：模型托管端点单独加 token（后来撤销）

本设计最初给模型托管端点加了一个专用凭据 `DH_V1_API_TOKEN` / `X-V1-Token`，
理由是「它是第一批给远程客户端调用的端点（桌面宠物指向远程部署）」。

**这条后来撤销了**：桌宠搬出去成了自己的应用，不再调用本服务，于是托管端点
只剩管理台一个调用方 —— 而管理台与服务同源，鉴权既没必要又碍事（服务端一配令牌，
管理台就 401，而界面上没有地方能填它）。令牌机制整体删除，两个面都不带凭据。

剩下的两件事**仍然成立**：

- Cubism 的**运行时**端点（`Core/` 与 `Shaders/`）不鉴权：core 经 `<script>` 标签加载、
  shader 由 framework 自行 fetch，两者都带不了请求头。
- 不给 `/api/v1/sessions`、`/api/v1/ws/*` 补鉴权。那是既有的、独立的安全问题，
  把它混进来会改到现有调用方。**应作为单独一件事处理**。

### 4.4 情绪标签

情绪从 LLM 输出里带出来，而不是另起一个分类模型。约定用行内标记：

```
[smile]好的呀，[think]让我想想……[/think]
```

SSE 的 delta 原样透传，**由客户端解析**并剥离标记后再显示 —— 服务端不维护对话历史、不做二次处理。这样表情目录是纯前端的关注点，改表情不用动服务端。

标记集刻意保持很小（`smile` / `sad` / `think` / `surprise` / `neutral`），且必须与模型 `config_json.expressions` 的 key 对齐。**未识别的标记一律落到 `neutral`**，不得抛错。

## 5. 渲染层设计（`dh-admin-ui/src/live2d/`）

> 本节描述的算法（包络、生命感、参数解析、帧序、表情）全部保持不变，只是**代码位置**在 `dh-admin-ui/src/live2d/` 下，并由 §5.6 的组件封装暴露给管理台。

### 5.1 口型驱动：离线预算包络，实时只做查表

**不**用实时 `AnalyserNode` 分析。TTS 返回的 WAV 在客户端解成 `AudioBuffer` 时，一次性扫出包络数组（约 **60 点/秒**），渲染时按 `currentTime − 播放起点` 查表并线性插值。

比实时分析好在三点，都很实际：

1. **确定性** —— Electron 与浏览器算出同一个值；不依赖音频图是否真的在跑（暂停、丢帧、被遮挡都不会让嘴失真）。
2. **可归一化** —— 每段音频按自身峰值归一化。否则音量偏小的句子嘴几乎不动。
3. **平滑质量更好** —— 非对称单极滤波（**开嘴快、闭嘴慢**）离线做，比逐帧凑平滑自然得多。

`ParamMouthForm`（张口 / 撮口形）用**零交叉率**推：最便宜，且与口型开合相关。

> ⚠️ **参数名与取值范围每个模型都不同。** `ParamMouthOpenY` 可能是 `0..1`、`-1..1`、也可能是 `0..30`。必须在装载时从模型的参数规格里读出实际范围再映射，**写死 `0..1` 在大多数真实模型上是错的**。个别模型根本没有该参数，需要降级路径（嘴不动，表情仍在）。

说话开始时另起一个包络基线（静音时 `ParamMouthOpenY = 0`），不做淡入淡出 —— TTS 音频本身首尾已有静音。

### 5.2 生命感层

| 层 | 参数 | 做法 |
|---|---|---|
| 呼吸 | `ParamAngleZ`（+ 轻微 `AngleX/Y`） | 正弦，周期约 4s；**说话时压低幅度**，避免与口型打架 |
| 眨眼 | `ParamEyeLOpen` | 每 ~4s 一个眨眼槽位，槽内按概率眨眼，偶发双眨 |
| 眼球扫视 | `ParamEyeBallX/Y` | 每 ~1.2s 换一个扫视目标（实现里是 1200 ms 一个槽位） |
| 注视光标 | `ParamEyeBallX/Y` | 桌宠专用：Electron 主进程轮询光标位置（窗口默认穿透鼠标，收不到 move 事件，只能轮询），渲染层按「距离越远、眼睛转得越多」平滑过去。有光标时取代扫视，没有时回到扫视 |
| 头部微动 | `ParamAngleX/Y/Z` | 多层不同周期的正弦叠加，慢、小幅度 |

**幅度是「模型自身量程的一个比例」，不是绝对值。** 各家 rig 的量程差得远（Natori 的 `ParamAngleZ` 是 -30..30，有的模型是 -1..1），写死数字等于在大多数模型上把头甩断、把头冻住。头动取量程的 ~10-15%，眼球扫视与注视取眼动量程的比例。映射在 `render/params.ts` 的 `offsetInRange`。

**关键**：头动参数写完后 Cubism 的 physics 会自动跑出头发/布料的跟随，所以不要试图自己算二级动画，也不要绕过 physics。

### 5.3 表情导演

emotion 标记 → `config_json.expressions` 映射 → 表情索引，由 Cubism 做交叉淡入。

**Cubism 5 没有 `SetExpression`** —— 表情是排队的 motion（`CubismExpressionMotionManager`
继承 `CubismMotionQueueManager`），套用表情 = 把它的 motion 入队，由管理器做交叉淡入。

三条实现要点：

- **入队只在表情变化时做。** `startMotion` 每次都往队列里 push 一项，逐帧调用会让队列无限增长。
- **下标 = 模型 `FileReferences.Expressions` 数组里的位置**，与 `dh-server/dh_server/live2d/registry.py` 校验口径一致。
- **没有表情 = 索引 -1**，调用 `stopAllMotions()` 回到模型本来的脸。

**中性不是「第 0 个表情」。** 真实模型的第 0 个表情是作者随手放的那个（Natori 的第 0 个是
`Angry`），沿用它会让一句没有情绪的话变成一张愤怒的脸。中性取运营自己映射的 `neutral`；
没有映射就不套任何表情。

表情类型必须**限定在眼睛 / 眉毛 / 脸颊**，避免携带嘴部参数 —— 否则会与 §5.1 写的口型直接冲突（见 §5.4 的帧序说明）。

### 5.4 帧序：Cubism 集成最易翻车处

每帧严格按此顺序：

```
1. model.loadParameters()       ← 基线复位成模型默认值（见下）
2. motionManager.Update()       ← 模型自带动作
3. expressionManager.Update()   ← 表情叠在动作之上
4. 我们写 mouth + idle 参数      ← 我们后写，所以赢得冲突
5. physics.Update()             ← 由头部参数派生头发/布料
6. pose.Update()                ← 必须在烘焙之前，否则晚一帧
7. model.Update()               ← 参数烘进顶点
8. draw()
```

四条铁律：

- **第 1 步不能省。** Cubism 的表情大多是**加性**的（Natori 的每一个都是 `blend: Add`），框架把它加到参数的**当前值**上。不复位的话，上一帧留下的值就是这一帧要加的基数，一次脸红会 0 → 1 → 2 → 3 地涨到脸都烂掉。`loadParameters()` 把参数复位成模型默认值（`loadModel` 时 `saveParameters()` 存过一次，之后我们不再存，所以基线恒为默认值），顺带让动作的淡出落回静止姿势，而不是上一帧恰好停在哪儿。
- **关闭 Cubism 的自动 breath / eyeblink / lipsync。** 两边同时写同一参数必然抖动。统一关掉，全靠自己驱动（§5.2 已经自己做了）。加载动作时必须 `setEffectIds([], [])` —— 框架在更新动作时**无保护地解引用**这两个列表，不给就是空指针；给空则是刻意的：它们是动作里 EyeBlink / LipSync 曲线的落点，而眨眼与口型都归我们自己驱动。
- **第 4 步必须在 Cubism 的 manager 之后。** 否则表情里带的参数会覆盖我们写的口型。
- **动作在前、表情在后**（与 SDK Demo 一致）：表情是「脸」，动作是「动画」，脸应该叠在动画上。

**我们的写入是「叠加」而不是「覆盖」。** 头动与眼球是**加性**（当前值 + 我们的偏移），眼睛开合是**乘性**（当前值 × 眨眼系数），只有口型是**绝对值**。这样表情把眼睛眯起来时宠物仍然会眨眼，模型自带动作的头动也仍然看得见；口型绝对值是 §5.4 的老规矩 —— 嘴归我们，表情不许碰。

### 5.5 生命周期

- **加载**：`model3.json` → 动作组 → 表情 → 参数规格（读出实际取值范围）。
- **待机**：无对话时只跑 §5.2 的生命感层。
- **说话**：句子队列驱动，逐句 TTS → 解码 → 起播。
- **呼吸相位必须由自由运行的时钟驱动，绝不按句重启** —— 否则每句话开头都会出现一次可见的相位跳变。句子边界只影响呼吸的**幅度**（说话时压低，见 §5.2），不影响相位的连续性。同理，头部微动的多层正弦全程不重置。

### 5.6 管理台集成：`<Live2DStage>` 组件

一个组件，管理台三个页面共用。props 只有「形象（avatar）」和「是否说话 + 当前音频包络」；组件内部负责装载、渲染循环、idle 层、把包络喂给 `applyFrame`。

**页面各自的接法**：

| 页面 | 接法 |
|---|---|
| 形象管理 | 不挂载组件。列出 `drive_model=live2d` 的形象，提供上传表单（zip + 名称 + 表情映射），上传即校验 |
| 会话调试 | 顶部形象选择器（现有下拉框按数字人模型分组）。选中 Live2D 时**不建 WebRTC session**，改为挂载 `<Live2DStage>` + 现有那个音频文件输入 |
| 数字人会话 | 同上。选中 Live2D 时走 `/api/admin/tts`，TTS 音频解码后既送去播放，也送去算包络 |

**关键分叉点**：现有 Debug / Chat 页在 `start()` 里建 WebRTC 连接、拉视频轨。Live2D 形象下这些一步都不能做 —— 服务端根本没有为 Live2D 建 session（§3.4）。所以分支要落在**建连接之前**，不是渲染阶段之后。

**配置落在系统 Tab**：

| 配置项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| Live2D 开关 | `DH_LIVE2D_ENABLED` | `true` | **仅上报**：只出现在 Live2D 健康检查的 JSON 里（系统页据此显示「启用：否」），服务端没有任何端点按它分支 |
| SDK 路径 | `DH_LIVE2D_SDK_DIR` | `data/CubismSdkForWeb-5-r.5` | Core / Shaders 所在目录；**服务端只从这里读**，经 `/api/v1/live2d/runtime/{path}` 下发给浏览器，不渲染任何 Live2D |
| 构图 | 存在形象 `config_json` | `scale=1, y=0` | 每个模型单独的缩放与垂直偏移，因为各模型原始构图差异很大 |

SDK 是**前端**资源，但路径存在服务端配置里 —— 因为管理台要能读到它才知道去哪儿取，并且能按部署环境改。服务端提供端点把 Core / Shaders 下发给浏览器（`/api/v1/live2d/runtime/{path}`，**不鉴权**：core 经 `<script>` 标签加载、shader 由 framework 自行 fetch，两者都带不了请求头），模型文件另走带 token 的托管端点。原先的 `public/cubism` 静态目录是调试期的临时做法，构建产物里不该有 SDK。

## 6. Electron 外壳

### 6.1 主进程四件事

| 事 | 关键点 |
|---|---|
| 无边框透明置顶窗 | `transparent` + `frame: false` + `alwaysOnTop` + `hasShadow: false` |
| 鼠标穿透 | `setIgnoreMouseEvents(true, { forward: true })`，收到 `mousemove` 后关掉穿透、移开后重新打开 |
| **关掉后台节流** | `webPreferences.backgroundThrottling: false` |
| 托盘菜单 | 服务地址、切模型、退出 |

### 6.2 后台节流必须单拎出来说

Electron 默认对**失焦或被遮挡**的窗口暂停 `requestAnimationFrame`。而桌面宠物**永远处于失焦状态** —— 它就是背景上的一个东西。

> **不加 `backgroundThrottling: false`，宠物会站在那儿一动不动。**

同类「必须在写代码之前就知道」的坑：

- 鼠标穿透依赖 `mousemove` 的重新命中检测，穿透期间拿不到事件
- 透明窗口在部分 Windows 缩放比例下出现黑边
- GPU 合成透明窗的开销：`app.disableHardwareAcceleration()` **需实测**，不要默认开

### 6.3 配置

服务地址走**环境变量 + 托盘菜单可改**，两处都要有。默认指向 `http://127.0.0.1:8000`。

## 7. 错误处理与降级

**首要原则：任何一环出错，宠物必须还活着。** 桌宠的全部价值在于它一直在那儿。失败必须是无声的、降级的，不能白屏、不能静默消失。

| 故障 | 行为 |
|---|---|
| 模型加载 / 校验失败 | 注册时即拦截（§4.2）；客户端加载失败显示明确错误文案，**不进入空转动画** |
| SSE 断流 | 退避重连；**idle 层持续运行**，宠物照常呼吸 |
| TTS 失败 / 超时 | 跳过该句、继续队列、照常出字幕（表情标记已先到） |
| 句子里出现未知情绪标记 | 落 `neutral`，不抛错 |
| 模型缺 `ParamMouthOpenY` | 嘴不动，表情与生命感照常（§5.1 降级路径） |
| 服务完全不可达 | Electron 显示「未连接」，**idle 层仍然运行** |
| 收到 401 | 端点要求凭据而客户端没带。**提示配置错误并停止重试** —— 重试不会自愈，只是刷日志 |

## 8. 测试策略

**Python**（沿用现有 torch-free pytest）：

- `registry` 校验：缺 `model3.json`、`.moc3` 不存在、表情 id 不匹配 → 各一条
- `drive_model` 迁移的幂等性：新库跑一次、旧库跑两次

> ⚠️ **测试一律用 scratch 库，绝不碰 `data/digital-human.db`。** 本地管理台的 demo key 绑在真实库上，用真库验证会作废用户的 key。

**前端**（vitest）：

- 包络计算：纯确定性数学 —— 峰值归一化、非对称平滑、静音段归零
- emotion → expression 映射，含未知标记降级
- 帧序逻辑（保证 idle/mouth 在 manager 之后写入）

**渲染本体**：WebGL 不可单测。用 Playwright 在**真 Electron** 里验 —— 项目已装 `.playwright-mcp`。至少覆盖：模型转起来、嘴随音频动、idle 持续、窗口失焦仍动（节流那条）。

## 9. 外部依赖与风险

| 项 | 说明 | 应对 |
|---|---|---|
| **Cubism SDK for Web** | 专有 SDK，**不走 npm**：Core 与 Shaders 由服务端从 `DH_LIVE2D_SDK_DIR` 读出、经 `/api/v1/live2d/runtime/{path}` 下发给浏览器，构建产物里不含 SDK | SDK 需要账号 + License Agreement，免费档有附带条件。**先在最小 Electron + Cubism 里验证「SDK 可取到、模型能转」，再动服务端** —— 这是唯一会否证整个方案的外部未知 |
| 模型文件 | 需求方自行下载 | 不在本设计范围。假定为合法的 Cubism 工程（`model3.json` + `.moc3` + 纹理 + physics + motions + expressions） |
| Live2D 各家模型参数差异 | 参数名与取值范围不统一 | §5.1 已要求运行时从模型读规格，禁止硬编码 |

## 10. 实测参数范围

Hiyori 的真实取值，印证了 §5.1「禁止硬编码 0..1」是硬需求：

| 参数 | 真实范围 | 写死 0..1 的后果 |
|---|---|---|
| `ParamMouthOpenY` | 0 .. 1 | 恰好正确 |
| `ParamMouthForm` | **-2 .. 1** | 口型反相 |
| `ParamAngleX/Y/Z` | **-30 .. 30** | **头部几乎不动** |
| `ParamEyeLOpen` | 0 .. 1.2 | 轻微过冲 |
| `ParamEyeBallX/Y` | -1 .. 1 | 恰好正确 |

七个受驱动参数里只有两个是常规值。