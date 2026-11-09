# 模型与运行环境

## 注册模型

| 模型 | 驱动方式 | 运行要求 | 形象素材 |
|---|---|---|---|
| MuseTalk 1.5 | 服务端驱动，WebRTC 视频 | NVIDIA GPU（推荐） | 照片 |
| Wav2Lip | 服务端驱动，WebRTC 视频 | CPU 可运行；GPU 更快 | 人脸清晰的口播视频（或照片） |
| Live2D | 客户端渲染 | 浏览器 WebGL；需 Cubism SDK | Cubism 工程 zip |

## MuseTalk

真实推理需要单独环境和权重：

```bash
./dh-server/scripts/setup_musetalk.sh
./dh-server/scripts/fetch_musetalk_weights.sh
python dh-server/run.py
```

MuseTalk 形象制作流程：上传照片 → 人脸检测/对齐/遮罩/VAE 编码，生成形象包。会话中按音频块驱动 UNet 并融合输出。

## Wav2Lip

Wav2Lip 使用通用权重，不按人训练。制作时会逐帧检测人脸并生成“闭嘴空闲轨”，避免源视频说话动作在空闲时回放。

```bash
./dh-server/scripts/setup_wav2lip.sh
```

CPU 环境示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[wav2lip]"
```

详细设计见 [Wav2Lip 引擎设计](superpowers/specs/2026-10-26-wav2lip-engine-design.md)。

## Live2D

服务端不渲染 Cubism 模型，只托管模型文件和 Cubism SDK 运行时资源。`DH_LIVE2D_SDK_DIR` 指向 SDK 目录；管理台/客户端负责加载与口型驱动。
