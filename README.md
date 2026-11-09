# 数字人服务（Digital Human Service）

实时交互数字人服务：接收音频与形象 ID，输出实时数字人视频流（WebRTC），并提供 React 管理台用于形象管理、会话调试和系统查看。

## 支持模型

| 模型 | 渲染位置 | 说明 |
|---|---|---|
| `musetalk` | 服务端 GPU | MuseTalk 1.5，输出 WebRTC 视频流 |
| `wav2lip` | 服务端 CPU/GPU | Wav2Lip，输出 WebRTC 视频流 |
| `live2d` | 客户端 WebGL | 服务端仅托管模型/SDK 文件，不产出像素 |

每个形象记录自己的 `drive_model`；模型权重可在管理台 Models 页加载/卸载。

## 快速开始

前置：Python 3.11+、Node 18+。MuseTalk 需要 NVIDIA GPU；Wav2Lip 可在 CPU 运行。


### 准备模型

- MuseTalk：按 [模型与运行环境](docs/models.md) 准备 `.venv-musetalk` 与权重后启动。
- Wav2Lip：执行 `./dh-server/scripts/setup_wav2lip.sh` 拉取源码与权重；CPU 环境可安装 CPU 版 torch 后使用 `pip install -e ".[wav2lip]"`。
- Live2D：配置 `DH_LIVE2D_SDK_DIR` 指向 Cubism SDK 目录。

### 启动服务端

``` bash
# 1. 配置环境变量
cp .env.example .env
# 2. 安装服务端依赖（用于开发/测试）
pip install -e "dh-server/[dev]"
# 3. 启动服务
python dh-server/run.py
# 可选参数：--host --port --reload --log-level
```

### 启动前端
``` bash
cd dh-admin-ui
npm install
npm run dev
cd ..
```


## 开发

```bash
cd dh-server && pytest
cd dh-admin-ui && npm run dev   # Vite 代理 /api 到 127.0.0.1:8000
```

## 文档索引

- [架构说明](docs/architecture.md)：工程结构、核心组件、渲染/驱动模型。
- [API 文档](docs/api.md)：`/api/v1`、`/api/admin`、会话指标、WebSocket 通道。
- [配置说明](docs/configuration.md)：`DH_*` 环境变量与非环境变量配置项。
- [模型与运行环境](docs/models.md)：MuseTalk / Wav2Lip / Live2D 的准备与注意事项。
- [设计规格归档](docs/superpowers/specs/)：历史设计文档（按需查阅）。

## 说明

- `/api/v1` 与 `/api/admin` 默认无鉴权；仅建议在受控环境使用，暴露到可达网络前请自行增加网关或认证。
- 技术栈：Python 3.11+、FastAPI、aiortc、SQLAlchemy + SQLite；React + Vite + TypeScript。
