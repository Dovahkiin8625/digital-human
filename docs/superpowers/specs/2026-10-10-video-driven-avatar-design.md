# 基于视频的形象制作 — 设计文档

日期：2026-10-10
状态：已与需求方对齐（视频用途 / 时长上限 / 循环方式 / 前端范围 / 开放入口均经确认）
关联：`docs/superpowers/specs/2026-10-08-digital-human-service-design.md`

## 1. 目标与范围

让「制作形象」在**单张照片**之外支持**上传短视频**。视频作为**动态驱动源**：会话中头部、身体、背景沿用视频里的自然运动，只有嘴部/下颌区域被实时音频重新生成——即 MuseTalk 的固有形态「视频配音」。照片仍是特例（只有一帧，头部静止）。

**入口**：租户 API `POST /api/v1/avatars` 与后台控制台 `POST /api/admin/avatars` 同时开放，两者共用同一套存储与流水线。

**改动落点**：形象包格式、`AvatarPipeline`、`MuseTalkEngine`、`AvatarStore`、上传校验、后台控制台的上传控件。会话协议、信令、WebRTC 出口、`RenderSession` 状态机均不变。

### 对原设计非目标的修订

原设计文档 §1「非目标」列有「真人视频驱动不做」。本次将其**收窄**为「不做 3D 数字人；真人视频驱动改为本设计的核心能力」。其余非目标不变。

### 非目标（YAGNI）

- 不做视频裁剪/选帧 UI：视频整体处理，超过上限直接报错，由用户自行剪短。
- 不做逐帧全身姿态跟踪：仍只用媒体管道的人脸关键点，沿用现有 `mediapipe` 路线，不引入 DWPose/mmpose。
- 不做「放完定格」「正向循环」等其它循环策略：只做乒乓循环。
- 不做磁盘帧窗口/按需流式加载：≤15s 的帧全部常驻内存。
- 不使用视频自带的音轨：嘴型始终由会话传入的音频驱动。

## 2. 背景：现状与差距

现状（见 `server/engine/musetalk_engine.py`）：

- `_AvatarState` 持有**单帧** `base_rgb`（720×1280 画布）+ **单个** `latents`（`[1,8,32,32]`）+ 单组 `bbox`/`mask`/`mask_crop_box`。
- `_render` 把 UNet 解出的下半脸融合回这张固定底图；`_device_assets` 把底图/掩码**一次性**搬到显存。
- `RenderSession._static_frame()` 读 `normalized.png` 作为空闲保活帧。

MuseTalk 参考实现（`vendor/MuseTalk/scripts/realtime_inference.py`）在视频下用三条**乒乓序列**：

```
frame_list_cycle        = frame_list + frame_list[::-1]
coord_list_cycle        = coord_list + coord_list[::-1]
input_latent_list_cycle = latent_list + latent_list[::-1]
# 第 i 帧取 [i % len(cycle)]
```

差距即：底图、bbox、掩码、latents 都要从「1 个」变成「N 个的循环」。

## 3. 关键决策

1. **统一为「帧循环」（方案 A）**。形象包一律是 N 帧的循环，照片是 N=1 的特例。引擎按输出帧序号索引循环。避免为图片/视频维护两条渲染路径；旧包以 N=1 兼容加载。
2. **乒乓循环**。`cycle_index(f, n)`：`n==1 → 0`；否则 `p = 2n-2`，`k = f % p`，返回 `k < n ? k : p-k`。首尾自然衔接，无跳变。
3. **不存解码后的帧**。磁盘只存**上传的原始视频**（已作为 `source<ext>` 保存）与小型 sidecar；底帧在**加载形象时**用与流水线**完全相同**的解码函数重建。磁盘从 ~1GB/形象降到几 MB。
4. **≤15 秒、超限直接报错**。上限在**上传时**校验并返回 422，不进入后台任务。
5. **目标帧率 = `settings.fps`（25）**。解码时按时间戳重采样到目标帧率，动作按自然速度播放，与源视频帧率无关。

## 4. 形象包格式

### 4.1 磁盘布局（视频形象，N 帧）

```
data/avatars/<avatar_id>/
  source.mp4                     # 上传原文件（沿用现有 source<ext> 命名）
  normalized.png                 # 第 0 个有效帧的画布（720×1280）：空闲保活帧 + 预览 + 旧格式兼容
  pack.json                      # {"musetalk": {...}}
  musetalk/
    latents.pt                   # [N,8,32,32] fp16，逐帧 VAE latent
    coords.json                  # [{"bbox":[x1,y1,x2,y2], "mask_crop_box":[..]}, ...]，长度 N
    masks/00000000.png …         # 逐帧融合掩码（bisenet，尺寸 = 该帧 mask_crop_box）；编号为有效帧序号 0..N-1
    meta.json                    # version/crop/detector/fps/source_fps/frame_count/mode/kept（kept=解码序列中被采用的有效帧下标）
```

照片形象（N=1）写入**同一结构**：`source.png`、单行 `coords.json`、单个 `masks/00000000.png`、`latents.pt` 形状 `[1,8,32,32]`。不再单独产出 `bbox.json` / `landmarks.json` / `mask.png`（旧包仍可读，见 §8）。

### 4.2 `pack.json` 的 `musetalk` 块

```json
{
  "musetalk": {
    "mode": "video",
    "dir": "musetalk",
    "version": "1.5",
    "crop": 256,
    "detector": "mediapipe.face_landmarker",
    "frame_count": 375,
    "fps": 25,
    "source_fps": 30.0,
    "base_image": "normalized.png",
    "source": "source.mp4",
    "latents": "musetalk/latents.pt",
    "coords": "musetalk/coords.json",
    "masks": "musetalk/masks",
    "bbox": [x1, y1, x2, y2],
    "mask_crop_box": [xs, ys, xe, ye]
  }
}
```

`bbox` / `mask_crop_box` 保留为**第 0 个有效帧**（`coords.json` 第一行）的值，供旧读取方与缩略图使用；引擎改读 `coords.json`。

## 5. 引擎改动（`server/engine/musetalk_engine.py`）

**`_AvatarState` 改为循环形态**

```
frames:      np.ndarray  [N,H,W,3] uint8   底帧，CPU，画布空间，sink 的 RGB 序
latents:     torch.Tensor [N,8,32,32]      常驻显存（N≤375 时约 6MB）
bboxes:      list[list[int]]  长度 N
masks:       list[np.ndarray] 长度 N       逐帧掩码，各帧尺寸不同（= 该帧 mask_crop_box）
mask_boxes:  list[list[int]]  长度 N
cycle_len:   int = N
```

删除 `base_t` / `mask_t` 的「建一次」缓存——底图与掩码现在逐帧不同。

**渲染（`_render`）**

- 对输出帧 `f = first + j` 取 `c = cycle_index(f, N)`，用 `latents[c]` 拼进本批的 UNet 输入（同一 avatar 的一批仍是**一次** UNet 调用：`torch.stack([latents[c] for 各帧])`）。
- 融合时底图取 `frames[c]`、掩码取 `masks[c]`、bbox 取 `bboxes[c]`。底帧按需 CPU→GPU 搬运（2.7MB/帧，~0.2ms），**显存不随 N 增长**。
- 分组逻辑不变：仍按 avatar 分组，组内共享一次 UNet 调用。

**加载（`_load_avatar`）**

- 读 `meta.json`：用 `kept` 从重建的解码序列里**选出有效帧**（流水线丢弃了无脸的帧，`kept[i]` 是第 i 个有效帧在解码序列中的下标），得到与 `coords.json` / `latents` 一一对应的 `frames`。
- **校验** `len(kept) == frame_count == latents.shape[0] == len(coords)`，且 `max(kept) < len(解码序列)`，否则报 `EngineUnavailable`（防止 coords/latents 与帧错位）。
- 读 `latents.pt`、`coords.json`、`masks/`。
- 旧格式回退：无 `coords.json` 时按单帧包加载（见 §8）。

**其它**

- `new_context` / `_advance` / `_drive_sync` 的帧序号推进逻辑不变——`ctx.emitted` 本就是绝对输出帧序号，正好作为 `cycle_index` 的输入。
- `RenderSession._static_frame()` 继续读 `normalized.png`（=第 0 帧），空闲保活无需改动。

## 6. 流水线与解码

### 6.1 解码函数（新增）

`server/avatar/video.py`：

```
decode_frames(path: Path, target_fps: int, max_frames: int)
    -> tuple[list[np.ndarray], float]   # (画布空间帧序列, source_fps)
```

- 用 `cv2.VideoCapture`（GPU venv 已有 cv2）按**时间戳**步进到 `target_fps`，保证自然速度且与源帧率无关。
- 每帧 letterbox 到 `frame_width × frame_height`（复用 `fit_to_canvas`），与现有照片路径的画布一致。
- 超过 `max_frames` 即停止并标记截断（由调用方决定报错）。
- 同一函数被流水线与引擎加载路径共用，保证两侧帧序列一致。

### 6.2 `MuseTalkAvatarPipeline.process`

1. 探测来源：图片 → 现有路径（N=1）；视频 → `decode_frames(target_fps=fps, max_frames=375)`。
2. 帧数 > `MAX_FRAMES(375)` → `PipelineError("video too long: …")`（正常在更早的上传校验就会拦下，此处为兜底）。
3. 逐帧：`landmarks` → `crop_box` → 256×256 裁剪 → `latents` → bisenet 掩码。**无脸的帧丢弃**并记日志。
4. 有效帧 < `MIN_FRAMES(8)` → `PipelineError("no usable face frames in video")`。
5. 写 `latents.pt`（堆叠）、`coords.json`、`masks/NNNN.png`、`meta.json`、`normalized.png`（第 0 个有效帧的画布）。
6. `progress()` 增量上报：`decode` / `detect_face` / `align` / `normalize` / `build_pack`，供后台页面显示进度。

`MockAvatarPipeline` 也接受视频（取第 0 帧 letterbox），保持 mock 路径可测。

## 7. 存储、校验与 API

**`AvatarStore`**：`save_source_image` 改名 `save_source`，按后缀分支——

- 图片：`PIL.verify()`，失败 → `invalid_image`（422）。
- 视频：cv2 探测，不可解码 → `invalid_video`（422）；时长 > `avatar_max_video_seconds` 或解码帧数 > `avatar_max_video_frames` → `video_too_long`（422）。

**在**上传时**就校验并返回 422**，不把注定失败的活排进后台任务。`Avatar.source_image_path` 列名保留（存来源文件路径），避免迁移。

**API**：路由不变。`POST /api/v1/avatars` 与 `POST /api/admin/avatars` 仍收 `name` + `file`，共用 `_create_avatar_rows`。仅放宽可接受后缀并加上述校验。新增错误码 `invalid_video`、`video_too_long`。

**前端**（`admin-ui/src/pages/Avatars.tsx`）：

- `Upload.Dragger` 的 `accept` 改为 `image/*,video/*`。
- 提示文案：「支持 JPG / PNG / MP4 / MOV，视频 ≤15 秒，建议清晰正脸半身」。
- 选中视频时做一次轻量时长检查（`<video>` 的 `duration`）做**即时**反馈；服务端仍是权威校验。

## 8. 兼容性

- **旧单帧形象包**（已有 `normalized.png` + `musetalk/{latents.pt, bbox.json, mask.png}`，无 `coords.json`）：`_load_avatar` 检测到后按 N=1 加载，`bbox`/`mask_crop_box` 取自 `pack.json` 的 `musetalk` 块。旧形象无需重建。
- **照片上传**：行为不变，只是产物写入新的统一结构。
- **Mock 引擎**：不受影响（不读 latents/sidecar）。

## 9. 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `DH_AVATAR_MAX_VIDEO_SECONDS` | `15.0` | 驱动视频时长上限，超出上传即 422 |
| `DH_AVATAR_MAX_VIDEO_FRAMES` | `375` | 解码帧数上限（15s × 25fps），与上式共同生效 |
| `DH_AVATAR_CACHE_SIZE` | `2` | 引擎按 avatar 缓存形象数；视频形象每份常驻约 1GB 内存，故默认由 4 降到 2 |

## 10. 错误处理

| 场景 | 处理 |
|---|---|
| 上传不可解码的视频 | 上传即 422 `invalid_video` |
| 视频 > 15s / 帧数超限 | 上传即 422 `video_too_long` |
| 视频全程无脸 / 有效帧 < 8 | 后台任务失败，`status=failed`，错误落库并在详情页可见 |
| 加载时重建帧数 ≠ `meta.frame_count` | `EngineUnavailable`，会话创建失败并给出明确信息 |
| 视频自带音轨 | 忽略；嘴型只由会话音频驱动 |
| 1 帧视频 / 源帧率非 25 | 正常处理（N=1；按时间戳重采样） |

## 11. 测试策略

**默认（torch-free）套件**

- `decode_frames`：以桩替换 cv2，断言时间戳重采样、letterbox 画布尺寸、`max_frames` 截断。
- `AvatarStore.save_source`：拒绝不可解码视频（`invalid_video`）、拒绝超长（`video_too_long`）、接受合规视频。
- API：mock 流水线下 `POST /api/v1/avatars` 与 `/api/admin/avatars` 接受视频文件。
- `cycle_index`：乒乓数学（n=1、n=2、n=3、越界回绕）。
- 引擎：以桩替换内部件，构造 N 帧 `_AvatarState`，断言每帧取到不同的底图/latents/bbox；断言旧单帧包仍加载。
- `MuseTalkAvatarPipeline.process` 契约：以注入的假解码器 + 假 preparer，断言 `latents.pt` 形状、`coords.json` 长度、`masks/` 数量、`meta.json` 字段、`normalized.png` 尺寸。

**GPU 标记套件（`-m gpu`）**

- 端到端：生成一段短视频 → 上传 → 制作 → 建会话 → 断言有帧输出，且相邻输出帧的底图内容随视频变化（非同一张静态底图）。

## 12. 内存与性能（已知取舍）

- **内存**：≤15s 的 720×1280 底帧约 1GB/形象常驻；缓存默认降到 2 份（§9）。不做磁盘帧窗口/流式加载（YAGNI），若后续形象变大再补。
- **显存**：latents 全量约 6MB/形象；底帧/掩码按批搬运，不随 N 常驻显存。
- **制作耗时**：375 帧 ×（检测 ~5ms + VAE 编码 ~15ms + 掩码 ~10ms）≈ 11s，另加一次性模型加载 ~11s，在后台任务中可接受。
- **磁盘**：仅存原始视频 + sidecar（latents ~6MB、掩码 ~数 MB），远小于 PNG 帧序列方案。
