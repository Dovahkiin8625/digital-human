# Live2D 数字人 — 设计文档

日期：2026-10-22
状态：已与需求方对齐（并存不替换 / 桌面瘦客户端 / v1 仅文字输入，三项均经确认）
关联：`docs/superpowers/specs/2026-10-08-digital-human-service-design.md`、`2026-10-10-video-driven-avatar-design.md`

## 1. 目标与范围

在**同一个服务**内增加 Live2D 数字人形态，与现有的真人视频数字人（MuseTalk / LivePortrait / Wan 2.2）**并存**。

一个交付物同时覆盖两个场景：

- **网页版** —— FastAPI 挂载 `/live2d`，浏览器直接访问。
- **桌面宠物** —— Electron 外壳加载同一份 bundle，渲染层本地、服务端远程。

服务端的对话能力（LLM / TTS / 鉴权 / 管理台）完全复用现有实现。

### 1.1 非目标（YAGNI）

- **不做语音输入（ASR）**。v1 仅文字。服务里没有 ASR 代码（`server/` 下的 whisper 引用是 MuseTalk 的音频特征提取器，不是识别器），引入它等于新增一个子系统。
- **不替换、不改动现有视频路径**。MuseTalk / LivePortrait / Wan 2.2 的引擎、制作流水线、WebRTC 出口、信令协议一行不改。
- **不给 Live2D 开新的 session 类型**。理由见 §3.2。
- **不做主动搭话 / 服务端推送 / 空闲时的宠物发言**。同理，随 session 一起去。
- **不碰服务端渲染的任何代码路径**。Live2D 不产出视频帧，因此不进入 `DriveEngine`。

### 1.2 与现有设计的关系

原设计 §1 把「服务边界」定为「音频驱动 + 渲染 + 视频流输出」，且「LLM/ASR 由调用方自建」。此后 `server/llm/` 与 `server/tts/` 已经收回服务内，`admin-ui` 的 Chat 页也已跑通「LLM → 句子队列 → TTS → 驱动」全链路。本设计顺着这个已成事实的边界继续，**不修订**原设计的任何一条。

## 2. 背景：现状与差距

**Live2D 与现有服务的根本差异**：Live2D 是**客户端渲染**。Cubism SDK 在浏览器 / Electron 里用 WebGL 绘制，服务端一个像素都不产出。服务端从「渲染视频」缩成「在语义粒度上给出时机与情绪」。

这个差异决定了本设计的大部分内容。

**现状差距**（已核实）：

| 现状 | 差距 |
|---|---|
| `Avatar` 表无 `kind` 列（`server/db/models.py:20`） | 无法区分两种形象形态 |
| LLM / TTS 只挂在 admin 面：`/api/admin/llm/chat/stream`、`/api/admin/tts` | 产品客户端用 `X-API-Key` 打不到，必须补业务面端点 |
| 无 Live2D 模型注册与托管 | 需要新增 |
| `admin-ui` 是 admin 鉴权的调试台 | 产品客户端鉴权与生命周期都不同，不宜混入 |
| **`/api/v1/*` 目前完全无鉴权。** README 关于 `X-API-Key` 的描述已过时 —— 鉴权代码在 no-tenant 重构中被删除，`server/api/admin.py:100` 明写「no auth dependencies, all endpoints public」 | 桌面宠物是**第一个连远程服务的客户端**，且新建的 LLM/TTS 端点直接消耗上游额度，必须单独鉴权。见 §4.3 |

**已有的有利条件**：

- 迁移是手写的：`_SCHEMA_MIGRATIONS`（`server/db/session.py`）是 `(table, column, ddl)` 元组，`init_db()` 每次幂等执行。加字段只需加一行。
- `Avatar.config_json` 是 JSON 列，Live2D 专属字段有现成落点。
- `server/llm/openai.py`、`server/tts/minimax.py` 已是 provider 无关实现，补业务面端点是路由层工作，不是重写。
- TTS 音频由服务端产出、客户端解码播放，这条链路 `Chat.tsx` 已跑通（MiniMax 上限 44100Hz vs 服务 48000Hz，重采样在浏览器侧完成）。

## 3. 关键决策

### 3.1 在同一个服务内做，不新开服务

现有服务真正的资产不是 MuseTalk，是**对话脑 + 语音管线 + 租户鉴权 + 会话状态机**。新开服务要么重写、要么复制这些 —— 复制的是最贵的部分。

同时注意：现有 `DriveEngine.drive() -> list[DriveFrame]` 返回 numpy 帧数组，**Live2D 不产出帧**。把它塞进这个接口要么写一个把参数序列伪装成假帧的适配器（纯浪费），要么在旁边开一条平行传输通道 —— 后者本质上是本设计，只是名字不同。所以选择**共用大脑、分开渲染路径**：会话与视频通道并列存在，互不干涉。

### 3.2 口型完全在客户端；v1 不开 session

**口型不进服务端。** 客户端已从 TTS 拿到 WAV 的 `AudioBuffer` 并正要送进扬声器，驱动嘴形的就是**同一个音频节点**。因为读的是正在发声的那个源，口型与声音**天然零漂移**。

替代方案（服务端做音素对齐后推参数流）会引入一个与实际播放不同步的时钟，恰好是这个方案要避免的。而且现有链路里音频由服务端合成，服务端推的时长未必等于客户端解码播放的实际时长 —— **代价高，收益为负**。

**因此 v1 不需要 session。** 文字输入下客户端自足：模型文件来自服务端，LLM 与 TTS 是普通 HTTP/SSE 调用，没有任何东西需要服务端主动推。真要会话级指标、主动搭话、断线重连，那是 v2，届时加一条 cue 通道即可，现在不必留疤。

### 3.3 一个 bundle，两种外壳

`live2d-ui/` 独立 Vite app，构建产物同时供给 FastAPI 挂载（网页版）与 Electron 打包（桌面版）。渲染代码只有一份。

Electron **不加载远程 URL**，而是加载本地 bundle、服务地址作为配置项。启动瞬时、无白屏闪烁、服务重启期间宠物照常呼吸。

「瘦客户端」的含义是**只有渲染层在本地**：LLM、TTS、会话、鉴权全在远程服务端。Electron 里**不带 Python、不带 torch**。

### 3.4 一个 Live2D 形象不需要新的 session 模型

Live2D 不参与会话，模型字段与真人形象字段（bbox / crop / mask / prep 张量）零重叠。此时另开一张表的收益只有「字段不打架」，代价却是两套列表 / 删除 / 鉴权路径 —— 不划算。故并入 `avatars`，形态由 `kind` 区分。

## 4. 服务端设计

### 4.1 数据模型：一行迁移

`Avatar` 增加 `kind` 列，默认 `'photoreal'`。现有数据原地不动：

```python
# server/db/session.py 的 _SCHEMA_MIGRATIONS 增加：
("avatars", "kind", "ALTER TABLE avatars ADD COLUMN kind VARCHAR(16) NOT NULL DEFAULT 'photoreal'"),
```

遵循该元组既有的幂等模式（`has_column` 检查后跳过）。SQLAlchemy 模型上对应加 `kind: Mapped[str] = mapped_column(String(16), default="photoreal", nullable=False)`。

Live2D 专属字段（模型目录、`model3.json` 入口、默认表情、动作组、emotion→expression 映射）全部放已有的 `config_json`，不新增列：

```json
{
  "kind": "live2d",
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

新包 `server/live2d/`：

- **`registry.py`** —— 注册与校验。校验项：`model3.json`（或 `model.model3.json`）存在且可解析、被引用的 `.moc3` 与纹理文件均存在、`expressions` / `motions` 引用的 id 在模型里真实存在。**注册即验证**，不把错误留到客户端渲染时才炸。
- **`assets.py`** —— 模型文件的带鉴权静态托管。

> ⚠️ 模型文件**必须走带鉴权的端点下发**，不能裸静态暴露目录。模型是按租户/部署配置的资产，不是公开 CDN 内容。

### 4.3 业务面 LLM 与 TTS

现有 LLM / TTS 只在 admin 面上，产品客户端用 `X-API-Key` 打不到。补两个业务面端点，内部复用同一套 provider 实现（`server/llm/openai.py`、`server/tts/minimax.py`）：

| 端点 | 复用现有 | 说明 |
|---|---|---|
| `POST /api/v1/chat/stream` | `/api/admin/llm/chat/stream` | `text/event-stream`，每帧 `{"delta": ...}` / `{"done": true}` / `{"error": ...}` |
| `POST /api/v1/tts` | `/api/admin/tts` | `{"text", "voice_id"}` → `audio/wav`。沿用「服务端只套 WAV 头、重采样放浏览器」的既有做法，服务端不引入音频编解码依赖 |

#### 鉴权：这两个端点单独加 token

现有 `/api/v1/*` **没有任何鉴权**（§2）。这两个新端点**不跟随该现状**，理由是它们与其它端点性质不同：

- 它们是**第一批给远程客户端调用**的端点（桌面宠物指向远程部署）；
- 它们**直接消耗上游额度**（LLM tokens、MiniMax 合成时长），暴露即等于送钱。

做法：新增环境变量 `DH_V1_API_TOKEN`。留空 = 关闭鉴权（本地开发无需配置），非空时两个端点要求请求头 `X-V1-Token` 与之相等，否则 401。错误走既有 `AppError` 信封。

**刻意不做**：不给现有 `/api/v1/avatars`、`/api/v1/sessions`、`/api/v1/ws/*` 补鉴权。那是既有的、独立的安全问题（README 已与其不符），把它混进来会改到现有调用方并让本设计的 diff 失焦。**应作为单独一件事处理**。

**同时补** `GET /api/v1/avatars?kind=live2d` 的形态过滤（现有列表端点加一个可选查询参数）。

### 4.4 情绪标签

情绪从 LLM 输出里带出来，而不是另起一个分类模型。在现有 system prompt 里约定行内标记：

```
[smile]好的呀，[think]让我想想……[/think]
```

SSE 的 delta 原样透传，**由客户端解析**并剥离标记后再显示 —— 服务端不维护对话历史、不做二次处理。这样表情目录是纯前端的关注点，改表情不用动服务端。

标记集刻意保持很小（`smile` / `sad` / `think` / `surprise` / `neutral`），且必须与模型 `config_json.expressions` 的 key 对齐。**未识别的标记一律落到 `neutral`**，不得抛错。

## 5. 客户端设计（`live2d-ui/`）

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
| 眨眼 | `ParamEyeLOpen` | 随机间隔 2–6s，偶发双眨 |
| 眼球扫视 | `ParamEyeBallX/Y` | 每 1–3s 一次快速跳动；光标进入中心 40% 画布宽以内时切换为平滑跟随，移开后平滑退回扫视（不得瞬切） |
| 头部微动 | `ParamAngleX/Y/Z` | 多层不同周期的正弦叠加，慢、小幅度 |

**关键**：头动参数写完后 Cubism 的 physics 会自动跑出头发/布料的跟随，所以不要试图自己算二级动画，也不要绕过 physics。

### 5.3 表情导演

emotion 标记 → `config_json.expressions` 映射 → `ExpressionManager.SetExpression(index)`，由 Cubism 做交叉淡入。

表情类型必须**限定在眼睛 / 眉毛 / 脸颊**，避免携带嘴部参数 —— 否则会与 §5.1 写的口型直接冲突（见 §5.4 的帧序说明）。

### 5.4 帧序：Cubism 集成最易翻车处

每帧严格按此顺序：

```
1. expressionManager.Update()   ← Cubism 先铺表情/动作
2. motionManager.Update()
3. 我们写 mouth + idle 参数     ← 我们后写，所以赢得冲突
4. physics.Update()             ← 由头部参数派生头发/布料
5. model.Update()               ← 参数烘进顶点
6. draw()
```

两条铁律：

- **关闭 Cubism 的自动 breath / eyeblink / lipsync。** 两边同时写同一参数必然抖动。统一关掉，全靠自己驱动（§5.2 已经自己做了）。
- **第 3 步必须在 Cubism 的 manager 之后。** 否则表情里带的参数会覆盖我们写的口型。

### 5.5 生命周期

- **加载**：`model3.json` → 动作组 → 表情 → 参数规格（读出实际取值范围）。
- **待机**：无对话时只跑 §5.2 的生命感层。
- **说话**：句子队列驱动，逐句 TTS → 解码 → 起播。
- **呼吸相位必须由自由运行的时钟驱动，绝不按句重启** —— 否则每句话开头都会出现一次可见的相位跳变。句子边界只影响呼吸的**幅度**（说话时压低，见 §5.2），不影响相位的连续性。同理，头部微动的多层正弦全程不重置。

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
| 收到 401 | 服务端已配 `DH_V1_API_TOKEN` 而客户端没带或带错。**提示配置错误并停止重试** —— 重试不会自愈，只是刷日志 |

## 8. 测试策略

**Python**（沿用现有 torch-free pytest）：

- 业务面 `/api/v1/chat/stream`、`/api/v1/tts` 的鉴权、SSE 分帧、错误映射
- `registry` 校验：缺 `model3.json`、`.moc3` 不存在、表情 id 不匹配 → 各一条
- `kind` 迁移的幂等性：新库跑一次、旧库跑两次

> ⚠️ **测试一律用 scratch 库，绝不碰 `data/digital-human.db`。** 本地管理台的 demo key 绑在真实库上，用真库验证会作废用户的 key。

**前端**（vitest）：

- 包络计算：纯确定性数学 —— 峰值归一化、非对称平滑、静音段归零
- emotion → expression 映射，含未知标记降级
- 帧序逻辑（保证 idle/mouth 在 manager 之后写入）

**渲染本体**：WebGL 不可单测。用 Playwright 在**真 Electron** 里验 —— 项目已装 `.playwright-mcp`。至少覆盖：模型转起来、嘴随音频动、idle 持续、窗口失焦仍动（节流那条）。

## 9. 外部依赖与风险

| 项 | 说明 | 应对 |
|---|---|---|
| **Cubism SDK for Web** | 走 Live2D **私有 npm registry**，需要账号 + License Agreement，免费档有附带条件 | **先做最小 Electron + Cubism 渲染 demo 验证 SDK 可装、模型可转，再动服务端**。这是唯一会否证整个方案的外部未知 |
| 模型文件 | 需求方自行下载 | 不在本设计范围。假定为合法的 Cubism 工程（`model3.json` + `.moc3` + 纹理 + physics + motions + expressions） |
| Live2D 各家模型参数差异 | 参数名与取值范围不统一 | §5.1 已要求运行时从模型读规格，禁止硬编码 |

## 10. 交付顺序

分四步，每步都能独立验证。**第 1 步先于一切服务端代码**。

1. **SDK 可行性验证**（丢弃性 spike）—— 最小 Electron + Cubism，转一个真实模型。验证 SDK 能装上、模型能转、参数规格能读出。**不过就不继续。**
2. **渲染内核** —— `live2d-ui` 骨架：模型装载、帧序、idle 层、一个写死参数的调试页。此阶段模型从**本地磁盘静态加载**（绕开 §4.2 的鉴权端点），以便在服务端就绪前独立调渲染。
3. **服务端** —— `kind` 迁移、registry、业务面 LLM/TTS、情绪标签。
4. **接线 + 外壳** —— 接上真实 LLM/TTS 链路，再套 Electron 托盘/穿透/置顶。

顺序刻意如此：把唯一可能否证方案的未知（§9）放在最前面，而把确定能用的部分（现有 LLM/TTS/鉴权）放在后面。