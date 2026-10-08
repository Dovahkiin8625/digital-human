# 商用数字人服务 — 设计文档

日期：2026-10-08
状态：已与需求方对齐（形态/边界/流媒体/算力路线均经确认）

## 1. 目标与范围

构建商用数字人服务，对外提供**实时交互数字人视频流**，并提供**后台调试页面**（形象制作、数字人调试）。

**产品形态**：实时交互数字人（2D 照片驱动）。
**服务边界**：只做「音频驱动 + 渲染 + 视频流输出」。LLM/ASR 由调用方自建；服务 API 收**音频 + 形象 ID**，出口是 WebRTC 视频流。
**算力**：自有 GPU 自部署口型驱动推理；引擎层可插拔，开发环境用 Mock 引擎。

### 非目标（YAGNI）

- 对话链路（ASR/LLM/TTS）不做
- 多 GPU 集群调度、SFU 分发不做（预留抽象，二期演化）
- 3D 数字人、真人视频驱动不做
- 计费/结算不做（仅留 Tenant 模型与用量指标打点）

## 2. 总体架构

方案 A（已确认）：**单体 GPU 渲染节点 + 内嵌 WebRTC**（FastAPI + aiortc 单进程）。

```
客户端(App/Web) ──REST──► API 层
       │                   │
       ├──WS 信令────────► 信令/媒体(aiortc) ──► RenderSession ×N
       │                                        ├ AudioIngest（RTP 或 WS 音频）
       └──WebRTC 媒体──────────────────────────►├ DriveEngine（Mock | MuseTalk）
                                                └ StreamSink（WebRTC video track）
后台调试台(React) ──REST/WS──► Admin API / AvatarStore（SQLite + 磁盘资产）
```

**组件职责**

| 组件 | 职责 | 接口 |
|---|---|---|
| API 层 | 形象 CRUD、会话生命周期、API Key 鉴权 | REST `/api/v1/*` |
| SessionManager / RenderSession | 会话编排与状态机（idle/listening/speaking/closed） | 信令 WS + WebRTC track |
| DriveEngine | 口型驱动引擎抽象：`drive(avatar_pack, audio_chunk) -> frames` | Python 接口，`mock`/`musetalk` 实现 |
| AvatarStore | 形象包管理：元数据（SQLite）+ 资产文件 | 内部 API |
| Admin 调试台 | 形象制作、会话调试、系统监控 | 复用 Admin API + WS 事件流 |

**关键决策**

1. 引擎可插拔是第一公民：`DRIVE_ENGINE=mock|musetalk`，无 GPU 可完整开发/测试/演示。
2. 形象制作 = 异步预处理任务（AvatarJob）：上传照片 → 人脸检测/对齐/裁剪归一化 → 形象包（归一化图 + 特征缓存 + 元数据）。Pipeline 接口同样 Mock/真实可切换。
3. 进程内 aiortc 而非独立 SFU；`StreamSink` 隔离媒体出口，未来可换推 SFU（方案 B 演化路径）。
4. 音频双通道进同一 `AudioIngest`：RTP audio track 为主，WS 音频分片为备；引擎无感知。
5. 会话短生命周期：显式 DELETE 或 60s 无音频自动关闭。

## 3. 数据模型

SQLite 起步（SQLAlchemy ORM，可平滑换 Postgres）。资产存 `data/avatars/{avatar_id}/`（路径可配置切 OSS）。

```
Tenant      id, name, api_key_hash, created_at
Avatar      id, tenant_id, name, status(pending|processing|ready|failed),
            source_image_path, pack_path, config_json, created_at
            -- config_json: 归一化参数、口型幅度、表情强度等引擎参数
AvatarJob   id, avatar_id, stage, progress, error, created_at
Session     id, tenant_id, avatar_id, status(idle|streaming|closed),
            started_at, closed_at, metrics_json
            -- metrics_json: 帧率、端到端延迟、丢帧率
```

## 4. 对外 API

`/api/v1`，鉴权 `X-API-Key`。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/avatars` | multipart 上传照片创建形象，返回 avatar_id，异步预处理 |
| GET | `/avatars` / `/avatars/{id}` | 列表/详情（status 供轮询制作完成） |
| DELETE | `/avatars/{id}` | 删除形象及资产 |
| POST | `/sessions` | 创建会话（avatar_id），返回 session_id + 信令地址 |
| DELETE | `/sessions/{id}` | 结束会话 |
| WS | `/ws/signal/{session_id}` | WebRTC 信令（SDP offer/answer、ICE） |
| WS | `/ws/audio/{session_id}` | 备选驱动通道：WS 推音频分片（二进制帧，PCM 20ms） |

媒体参数：音频 48kHz 16bit PCM，20ms 分帧；视频 25fps，720×1280，H.264（WebRTC video track）。

管理面 `/api/admin/*`：形象制作调试、会话调试、系统状态、租户与 API Key 管理（首期单管理员账号，Admin 密码登录）。

## 5. 后台调试台

React + Vite + TS SPA，挂 FastAPI 静态服务（单进程部署，无跨域）。

| 页面 | 能力 |
|---|---|
| 形象管理 | 列表/搜索、上传创建、预处理进度（AvatarJob）、状态徽标、删除 |
| 形象制作/调试 | 上传 → Pipeline 阶段展示 → 参数调优（口型幅度、表情强度、裁剪归一化）→ 保存形象包；静态预览 + 试驱动片段 |
| 会话调试 | 选形象建调试会话 → WebRTC 实时预览流；驱动音源三选一（麦克风/上传音频/假数据波形）；引擎参数热调；指标面板（FPS、延迟、丢帧）+ 事件日志 |
| 系统 | 引擎切换（mock/musetalk）、GPU 状态、租户/API Key 管理、任务监控 |

## 6. 错误处理

- 统一错误信封：`{"error": {"code", "message", "detail"}}`
- 形象制作失败：`status=failed` + error，支持重试
- 引擎运行时异常：会话置错并推 WS 事件，不拖垮进程
- `/healthz`：含引擎/GPU 状态
- 会话空闲 60s 自动回收；防资源泄漏

## 7. 测试策略

- MockEngine 使全链路无 GPU 可测
- 单测：引擎接口契约、形象管线、会话状态机、鉴权
- 集成：FastAPI TestClient（REST + WS）
- 人工 E2E：后台调试台走通「上传照片 → 制作 → 调试会话看流」
- TDD 推进

## 8. 技术栈与目录结构

Python 3.11 / FastAPI / aiortc / SQLAlchemy + SQLite；React + Vite + TS；生产引擎 MuseTalk（可插拔）。

```
digital-human/
├─ server/                  # Python 包
│  ├─ main.py               # FastAPI app factory
│  ├─ config.py
│  ├─ api/                  # REST + WS：avatars, sessions, admin, auth
│  ├─ core/                 # session_manager, render_session, audio_ingest, stream_sink
│  ├─ engine/               # base(接口), mock_engine, musetalk_engine
│  ├─ avatar/               # store, pipeline, mock_pipeline
│  ├─ db/                   # models, session
│  └─ admin/                # admin 鉴权与管理面逻辑
├─ admin-ui/                # React + Vite 调试台
├─ data/                    # 运行时资产（gitignore）
├─ tests/
├─ docs/superpowers/specs/
└─ pyproject.toml
```

## 9. 二期演化（预留，不做）

- Session/StreamSink 抽象 → 方案 B（网关 + GPU worker 池 + SFU）
- 文本输入 + 可插拔 TTS
- 计费、租户自助注册、多管理员 RBAC
