# 商用数字人服务 — 设计文档

日期：2026-10-08
状态：已与需求方对齐（形态/边界/流媒体/算力路线均经确认）

## 1. 目标与范围

构建商用数字人服务，对外提供**实时交互数字人视频流**，并提供**后台调试页面**（形象制作、数字人调试）。

**产品形态**：实时交互数字人（2D 照片驱动）。
**服务边界**：只做「音频驱动 + 渲染 + 视频流输出」。服务 API 收**音频 + 形象 ID**，出口是 WebRTC 视频流。ASR 仍由调用方自建；LLM 与 TTS 后来收回服务内（`dh-server/dh_server/llm/`、`dh-server/dh_server/tts/`），管理台据此提供「数字人对话」链路。
**算力**：自有 GPU 自部署口型驱动推理；引擎层可插拔，无 GPU 时可用纯 CPU 的 `wav2lip` 引擎。

### 非目标（YAGNI）

- 语音识别（ASR）不做：服务只收音频，不做识别（LLM 与 TTS 后来收回服务内）
- 多 GPU 集群调度、SFU 分发不做（预留抽象，二期演化）
- 3D 数字人不做（Live2D 属 2D）；真人视频驱动由 2026-10-10 的设计收窄进来，成为核心能力
- 计费/结算不做（仅留用量指标打点）

## 2. 总体架构

方案 A（已确认）：**单体 GPU 渲染节点 + 内嵌 WebRTC**（FastAPI + aiortc 单进程）。

```
客户端(App/Web) ──REST──► API 层
       │                   │
       ├──WS 信令────────► 信令/媒体(aiortc) ──► RenderSession ×N
       │                                        ├ AudioIngest（RTP 或 WS 音频）
       └──WebRTC 媒体──────────────────────────►├ DriveEngine（MuseTalk | wav2lip）
                                                └ StreamSink（WebRTC video track）
后台调试台(React) ──REST/WS──► Admin API / AvatarStore（SQLite + 磁盘资产）
```

**组件职责**

| 组件 | 职责 | 接口 |
|---|---|---|
| API 层 | 形象 CRUD、会话生命周期、业务面 API（目前无鉴权） | REST `/api/v1/*` |
| SessionManager / RenderSession | 会话编排与状态机（idle/listening/speaking/closed） | 信令 WS + WebRTC track |
| DriveEngine | 口型驱动引擎抽象：`drive(avatar_pack, audio_chunk) -> frames` | Python 接口，`musetalk`/`wav2lip` 实现 |
| AvatarStore | 形象包管理：元数据（SQLite）+ 资产文件 | 内部 API |
| Admin 调试台 | 形象制作、会话调试、系统监控 | 复用 Admin API + WS 事件流 |

**关键决策**

1. 引擎可插拔是第一公民：每个形象记录自己所属的**数字人模型**（`avatars.drive_model`），服务端按这个模型取引擎，多个模型可以同时驻留。引擎不再由环境变量切换（`DRIVE_ENGINE` 已移除），无 GPU 时可用纯 CPU 的 `wav2lip` 引擎跑通全链路。
2. 形象制作 = 异步预处理任务（AvatarJob）：上传照片 → 人脸检测/对齐/裁剪归一化 → 形象包（归一化图 + 特征缓存 + 元数据）。流水线与引擎在 `dh-server/dh_server/model_registry.py` 里成对注册（一个模型 = 一条流水线 + 一个引擎）。
3. 进程内 aiortc 而非独立 SFU；`StreamSink` 隔离媒体出口，未来可换推 SFU（方案 B 演化路径）。
4. 音频双通道进同一 `AudioIngest`：RTP audio track 为主，WS 音频分片为备；引擎无感知。
5. 会话短生命周期：显式 DELETE 或 60s 无音频自动关闭。

## 3. 数据模型

SQLite 起步（SQLAlchemy ORM，可平滑换 Postgres）。资产存 `data/avatars/{avatar_id}/`（路径可配置切 OSS）。

```
Avatar      id, name, status(pending|processing|ready|failed),
            drive_model, source_image_path, pack_path, config_json, created_at
            -- drive_model: 驱动这个形象的数字人模型（musetalk / wav2lip / live2d）
            -- config_json: 归一化参数、口型幅度、表情强度等引擎参数
AvatarJob   id, avatar_id, stage, progress, error, created_at
Session     id, avatar_id, status(idle|listening|speaking|closed|error),
            started_at, closed_at, metrics_json
            -- metrics_json: 帧率、端到端延迟、丢帧率
```

租户与 API Key 曾是设计的一部分，随多租户重构删除：现在 `/api/v1/*` 与
`/api/admin/*` 都无鉴权，唯一可选的凭据是 Live2D 模型托管端点的 `X-V1-Token`。

## 4. 对外 API

`/api/v1`。业务面目前**无鉴权**（`X-API-Key` 随多租户重构移除）；唯一可选的凭据是 Live2D 模型托管端点的 `X-V1-Token`（`DH_V1_API_TOKEN` 非空时才校验）。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/avatars` | multipart 上传照片创建形象，返回 avatar_id，异步预处理 |
| GET | `/avatars` / `/avatars/{id}` | 列表/详情（status 供轮询制作完成） |
| DELETE | `/avatars/{id}` | 删除形象及资产 |
| POST | `/sessions` | 创建会话（avatar_id），返回 session_id + 信令地址 |
| DELETE | `/sessions/{id}` | 结束会话 |
| WS | `/ws/signal/{session_id}` | WebRTC 信令（SDP offer/answer、ICE） |
| WS | `/ws/audio/{session_id}` | 备选驱动通道：WS 推音频分片（二进制帧，s16le 单声道 48kHz PCM） |

媒体参数：音频 48kHz 16bit PCM（引擎按 100 ms 一块消费）；视频 25fps，720×1280，H.264（WebRTC video track）。

管理面 `/api/admin/*`：形象制作调试、会话调试、数字人对话（LLM/TTS）、模型加载/卸载、系统状态（同样无鉴权）。

## 5. 后台调试台

React + Vite + TS SPA，挂 FastAPI 静态服务（单进程部署，无跨域）。

| 页面 | 能力 |
|---|---|
| 形象管理 | 列表/搜索、上传创建、预处理进度（AvatarJob）、状态徽标、删除 |
| 形象制作/调试 | 上传 → Pipeline 阶段展示 → 参数调优（口型幅度、表情强度、裁剪归一化）→ 保存形象包；静态预览 + 试驱动片段 |
| 会话调试 | 选形象建调试会话 → WebRTC 实时预览流；驱动音源三选一（麦克风/上传音频/假数据波形）；引擎参数热调；指标面板（FPS、延迟、丢帧）+ 事件日志 |
| 数字人对话 | 文字输入 → LLM → 句子队列 → TTS → 驱动 |
| 系统 | 数字人模型的加载/卸载、GPU 状态、运行时状态、任务监控 |

## 6. 错误处理

- 统一错误信封：`{"error": {"code", "message", "detail"}}`
- 形象制作失败：`status=failed` + error，支持重试
- 引擎运行时异常：会话置错并推 WS 事件，不拖垮进程
- `/healthz`：含各数字人模型的驻留状态与 GPU 状态
- 会话空闲 60s 自动回收；防资源泄漏

## 7. 测试策略

- 引擎以桩替换，全链路无 GPU 可测（`wav2lip` 引擎本身就是纯 CPU 的真实实现）
- 单测：引擎接口契约、形象管线、会话状态机、token 校验
- 集成：FastAPI TestClient（REST + WS）
- 人工 E2E：后台调试台走通「上传照片 → 制作 → 调试会话看流」
- TDD 推进

## 8. 技术栈与目录结构

Python 3.11 / FastAPI / aiortc / SQLAlchemy + SQLite；React + Vite + TS；生产引擎 MuseTalk（可插拔）。

```
digital-human/
├─ run.py                   # 启动入口（钉工作目录、import 路径、启动前体检）
├─ dh-server/dh_server/                  # Python 包
│  ├─ main.py               # FastAPI app factory
│  ├─ config.py
│  ├─ model_registry.py     # 数字人模型注册表：一个模型 = 一条流水线 + 一个引擎
│  ├─ api/                  # REST + WS：avatars, sessions, admin, live2d
│  ├─ core/                 # session_manager, render_session, audio_ingest, stream_sink, auth, errors
│  ├─ engine/               # base(接口), musetalk_engine, wav2lip_engine, pool
│  ├─ avatar/               # store, pipeline, musetalk_pipeline, wav2lip_pipeline, live2d_pipeline, video
│  ├─ db/                   # models, session
│  ├─ llm/  tts/            # OpenAI 兼容大模型 / MiniMax TTS
│  ├─ live2d/               # registry（模型校验）, archive（安全解压）
│  └─ wav2lip/              # Wav2Lip 的音频、几何、检测与包
├─ dh-admin-ui/                # React + Vite 调试台（含 Electron 桌宠外壳）
├─ scripts/                 # 环境与权重的准备脚本
├─ data/                    # 运行时资产（gitignore）
├─ tests/
├─ docs/superpowers/specs/
└─ pyproject.toml
```

## 9. 二期演化（预留，不做）

- Session/StreamSink 抽象 → 方案 B（网关 + GPU worker 池 + SFU）
- 计费、多管理员 RBAC

（文本输入 + 可插拔 TTS 已在此后的迭代中落地，见管理台的「数字人对话」页。）
