# API

媒体参数：音频 48kHz / 16bit / 单声道 PCM；视频 25fps、720×1280、H.264（WebRTC）。

> 两个 API 面默认**无鉴权**。部署到可达网络前请自行加网关/鉴权。

## 业务接口 `/api/v1`

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/sessions` | 创建会话，请求 `{"avatar_id"}`，形象须为 `ready` |
| GET | `/api/v1/sessions/{session_id}` | 会话详情（含 `metrics`） |
| DELETE | `/api/v1/sessions/{session_id}` | 结束会话（204） |
| WS | `/api/v1/ws/signal/{session_id}` | WebRTC 信令（SDP/ICE） |
| WS | `/api/v1/ws/audio/{session_id}` | 备用音频输入：二进制 s16le 单声道 48kHz PCM；文本 `{"type":"flush"}` 排空残留 |
| GET | `/api/v1/live2d/avatars/{id}/model/{path}` | 下发 Live2D 模型文件（仅模型格式） |
| GET | `/api/v1/live2d/runtime/{path}` | 下发 Cubism SDK `Core/` 与 `Shaders/` |

健康检查：`GET /healthz`。

## 会话指标 `metrics`

| 字段 | 含义 |
|---|---|
| `frames_out` | 音频驱动产生的帧数 |
| `keepalive_frames` | 空闲保活帧数 |
| `dropped_frames` | sink 已关闭后仍尝试推送的帧数 |
| `fps` | 名义帧率（配置值） |
| `avg_latency_ms` | 每块延迟 EMA |
| `blocks` | 已处理音频块数（每块 100ms） |
| `state` | `idle` / `listening` / `speaking` / `closed` / `error` |
| `queue_depth` | 待处理作业数（同步引擎恒为 0） |
| `overflows` | 视频轨队列满导致的丢帧数 |
| `p50_latency_ms` / `p95_latency_ms` / `max_latency_ms` | 最近窗口延迟分位数 |
| `latency_histogram` | 延迟分桶（零桶省略） |

每块承载 100ms 音频；块延迟长期超过 100ms 表示渲染跟不上输入，应结合 `overflows`、`queue_depth` 排查。

## 管理接口 `/api/admin`

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/admin/system` | 运行时/GPU/版本/数据目录 |
| GET | `/api/admin/models` | Models 页聚合状态（模型、服务、GPU） |
| GET | `/api/admin/drive-models` | 已注册数字人模型清单 |
| POST | `/api/admin/models/{model_id}/load` / `unload` | 加载/卸载模型权重 |
| POST | `/api/admin/services/{service_id}/load` / `unload` | 加载/卸载服务 |
| GET/POST | `/api/admin/backends`、`/api/admin/i2v` | 图生视频后端与任务提交 |
| GET/DELETE | `/api/admin/tasks`、`/api/admin/tasks/{task_id}` | 生成任务 |
| GET | `/api/admin/tasks/{task_id}/result.mp4` | 任务结果视频 |
| GET/POST/DELETE | `/api/admin/liveportrait-drivers*` | LivePortrait 动作模板 |
| GET/POST | `/api/admin/avatars` | 形象列表/创建（multipart：`name`、`model`、`file`；`live2d` 可选 `entry`/`expressions`；`wav2lip` 可选 `idle_crf`/`idle_preset`） |
| GET/DELETE | `/api/admin/avatars/{avatar_id}` | 形象详情/删除 |
| POST | `/api/admin/avatars/{avatar_id}/retry` | 重试制作 |
| GET | `/api/admin/avatars/{avatar_id}/idle.mp4` | 空闲循环视频（支持 Range） |
| GET | `/api/admin/avatars/{avatar_id}/normalized.png` | 静态画面 |
| GET | `/api/admin/avatars/{avatar_id}/artifacts` | 产物清单 |
| GET | `/api/admin/avatars/{avatar_id}/artifact/{path}` | 产物文件 |
| GET/PATCH | `/api/admin/avatars/{id}/live2d` | Live2D 表情/动作信息与映射 |
| GET | `/api/admin/live2d/health` | Cubism SDK 状态 |
| POST | `/api/admin/tts` | 文本合成 WAV（调试页使用） |
| GET/PUT | `/api/admin/tts/config` | MiniMax TTS 配置 |
| GET/PUT | `/api/admin/llm/config` | OpenAI 兼容 LLM 配置 |
| POST | `/api/admin/llm/chat/stream` | LLM 流式对话（SSE） |
| GET | `/api/admin/jobs` | 形象制作任务 |
| GET/POST/DELETE | `/api/admin/sessions*` | 管理会话与回收 |

管理台文本驱动调试：页面通过 `/api/admin/tts` 获取音频后，在浏览器解码并按 48kHz 通过 WS 音频通道驱动；WebRTC 仅回传视频轨，音频由页面本地回放。
