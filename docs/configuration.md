# 配置

服务端通过 `DH_` 前缀读取环境变量，也可写入项目根目录 `.env`（已 gitignore）。模板见 `.env.example`。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DH_DATA_DIR` | `data` | 运行时数据目录（SQLite、形象资产、模型权重） |
| `DH_DATABASE_URL` | `sqlite:///data/digital-human.db` | SQLAlchemy 数据库 URL |
| `DH_ADMIN_UI_DIR` | 空 | 管理台构建产物目录；空表示 `dh-admin-ui/dist` |
| `DH_SESSION_IDLE_TIMEOUT` | `60` | 会话无音频自动回收秒数 |
| `DH_SESSION_SWEEP_INTERVAL` | `10` | 空闲回收扫描周期（秒） |
| `DH_FRAME_WIDTH` | `720` | 输出视频宽 |
| `DH_FRAME_HEIGHT` | `1280` | 输出视频高 |
| `DH_FPS` | `25` | 输出帧率 |
| `DH_VIDEO_BITRATE` | `3500000` | 出站 H.264 目标码率（bps） |
| `DH_VIDEO_MAX_BITRATE` | `4000000` | 出站 H.264 码率上限（bps） |
| `DH_SAMPLE_RATE` | `48000` | 音频采样率（Hz） |
| `DH_AVATAR_MAX_VIDEO_SECONDS` | `15.0` | 驱动视频时长上限（秒） |
| `DH_AVATAR_MAX_VIDEO_FRAMES` | `375` | 驱动视频解码帧数上限 |
| `DH_AVATAR_MAX_UPLOAD_BYTES` | `300000000` | 上传字节上限 |
| `DH_AVATAR_CACHE_SIZE` | `2` | 引擎按形象缓存份数 |
| `DH_LIVE2D_ENABLED` | `true` | 仅用于健康检查/展示上报，不控制 Live2D 功能开关 |
| `DH_LIVE2D_SDK_DIR` | `data/CubismSdkForWeb-5-r.5` | Cubism SDK 目录（服务端仅下发 Core/Shaders） |
| `DH_LIVE2D_MAX_ARCHIVE_ENTRIES` | `512` | Live2D 压缩包条目数上限 |
| `DH_LIVE2D_MAX_UNPACKED_BYTES` | `512000000` | Live2D 解压后累计字节上限 |
| `DH_LIVE2D_MAX_ENTRY_BYTES` | `64000000` | Live2D 单文件解压字节上限 |

## 不通过环境变量配置的项

- **MiniMax TTS / OpenAI 兼容 LLM**：在管理台 Models/设置页写入 SQLite `settings` 表。
- **数字人模型选择**：每个形象记录自己的 `drive_model`；模型权重在 Models 页加载/卸载。

未知的 `DH_*` 变量会被忽略。
