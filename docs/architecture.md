# 架构

## 工程结构

仓库包含两个独立工程，通过 HTTP API 交互：

```text
digital-human/
├── dh-server/      # Python 服务端（FastAPI、引擎、SQLite、形象资产）
├── dh-admin-ui/    # React + Vite 管理台（构建产物由服务端挂载到 /admin）
├── data/           # 运行时数据（形象包、权重、SQLite、Cubism SDK）
└── docs/           # 文档
```

## 组件

- **API 层**：会话生命周期（`/api/v1/*`）、Live2D 模型托管、管理接口（`/api/admin/*`）。
- **SessionManager / RenderSession**：会话编排与状态机（`idle → listening → speaking → closed`）。空闲时推送保活帧，避免视频轨冻结。
- **DriveEngine**：口型驱动抽象 `drive(avatar_pack, audio_chunk) -> frames`。当前实现：`musetalk`（GPU）、`wav2lip`（CPU/GPU）。
- **AvatarPipeline**：上传素材 → 形象包。当前实现：`musetalk`、`wav2lip`。
- **模型注册表**（`dh-server/dh_server/model_registry.py`）：按模型成对注册流水线与引擎。已注册：MuseTalk、Wav2Lip（服务端驱动）、Live2D（客户端渲染，无服务端引擎）。
- **AvatarStore**：SQLite 元数据 + 磁盘资产。
- **音频输入**：WebRTC RTP 音频轨（主）与 WebSocket PCM 分片（备）进入同一 `AudioIngest`。

## 数字人模型

| 模型 | 渲染位置 | 说明 |
|---|---|---|
| `musetalk` | 服务端 GPU | MuseTalk 1.5，输出 WebRTC 视频流 |
| `wav2lip` | 服务端 CPU/GPU | Wav2Lip，输出 WebRTC 视频流；实时流中未到达的未来音频窗口用静音填充 |
| `live2d` | 客户端 WebGL | 服务端仅托管模型/SDK 文件；口型由客户端根据播放音频包络驱动 |

详细设计见 `docs/superpowers/specs/` 下对应规格文档。
