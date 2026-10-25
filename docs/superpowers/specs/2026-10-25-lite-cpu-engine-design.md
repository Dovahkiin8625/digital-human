# 轻量数字人引擎（lite / FeatherTalk）—— 设计文档

日期：2026-10-25
状态：已实现

---

## 1. 背景

服务里原本只有一个服务端驱动的数字人模型：MuseTalk 1.5。它是 GPU 专属的：

| | MuseTalk 1.5 | FeatherTalk（lite） |
|---|---|---|
| 参数量 | 947 M（VAE + UNet + Whisper） | **9 M**（音频 3.4 M + 视觉 5.5 M） |
| 权重体积 | ~4 GB | **22 MB** |
| 硬件 | 8–12 GB 显存 | **纯 CPU** |
| 一帧推理 | 4.5 ms（batch 8，GPU） | 17 ms（CPU，8 线程） |

也就是说，一个没有显卡的机器**完全跑不了 MuseTalk**：它的 UNet 是 fp16 的，
CPU 上 `Conv2d` 直接报错（`server/engine/musetalk_engine.py` 里 `.half()` 是
无条件调用的）。而"能不能在只有 CPU 的机器上提供数字人服务"是一个产品问题，
不是优化问题。

本设计要的是**第二个服务端驱动的模型**：轻到可以在 CPU 上实时跑。

## 2. 选型

候选与结论：

| 候选 | 结论 |
|---|---|
| **FeatherTalk**（anliyuan，Apache-2.0） | **采用**。唯一同时满足"权重拿得到 + CPU 实时 + 有人维护"的开源模型。仓库在 `vendor/FeatherTalk`，锁 commit。 |
| Ultralight-Digital-Human（FeatherTalk 的前身） | 同一作者，音频前端是 HuBERT-Large（1.2 GB，比 FeatherHuBERT 慢约 40 倍）。已被 FeatherTalk 取代。 |
| LiteAvatar（阿里达摩院，MIT） | 权重在 ModelScope（拿得到），但三段式流水线的前端是中文 ASR（Paraformer-large ~1 GB），形象来自预训练画廊而非任意照片。与现有产品形态冲突最大。 |
| CASync | 19.8 M / 4.08 GFLOPs，指标漂亮，但仓库只有 3 star、需按身份微调、无官方 ONNX。成熟度不足。 |
| Wav2Lip | 通用免训练，但只有 96×96 的嘴部像素 —— 正是 MuseTalk 要取代的东西。 |
| SadTalker / LatentSync / Hallo / EchoMimic / EMO | 全 GPU，显存 6–24 GB。出局。 |
| 纯几何/网格口型 | 零权重，但要和"照片级唇形"二选一；本轮按"要一个真模型"来做。 |

FeatherTalk 的代价是**个性化**：视觉 UNet 属于某个人，要用这个人自己的
视频训练。这意味着产品形态变了——不是"上传一张照片"，而是"上传一段这个人的
口播视频"。

## 3. 架构

```
制作端（GPU）                        服务端（CPU）
─────────────                        ───────────
上传 3~5 分钟口播视频
  │
  ├─ 解码 → 720×1280 画布帧
  ├─ SCRFD + PFLD → 每帧人脸框
  ├─ FeatherHuBERT → 训练用音频特征
  ├─ 训练专属视觉 UNet（5.46 M）  ──►  lite/visual.pth
  └─ 产出形象包：
       lite/frames/*.jpg   运动源帧（头/身体）
       lite/coords.json    每帧人脸框
       lite/visual.pth     专属视觉模型
       lite/meta.json
                                    │
                                    ▼
                             每 100 ms 音频块：
                               FeatherHuBERT（滚动窗口）→ 40 token
                               视觉 UNet → 144×144 脸块
                               贴回原帧 → 720×1280 RGB
```

一个模型 = 一条流水线 + 一个引擎（`server/model_registry.py`），本设计新增
`lite` 模型，流水线与引擎成对注册。

## 4. 关键决策

### 4.1 因果音频窗口（最重要的一条）

上游 FeatherTalk 的音频窗口是**以当前帧为中心**的 20 帧窗口 `[i-10, i+10)`，
包含 **9 帧未来（360 ms）**。用它渲染，画面显示的是"还没说出口的音"对应的嘴型 ——
在直播里就是**嘴比声音晚 360 ms**。浏览器拿服务端回传的音频轨做 A/V 时钟基准，
这个偏移肉眼可见。

本服务改用**因果窗口**：窗口里未来的 9 帧槽位填当前帧的 token。更重要的是，
**训练也用因果窗口**（`server/lite/dataset.py` 的 `CausalTemporalMouthRoiDataset`
只重写音频窗口，其余训练循环原样复用上游 `train.py`）——训练与推理分布一致，
不会出现"训练时看得到未来、推理时看不到"的错配。

代价：与上游预训练权重不兼容。但上游本来就没有可用的个性化权重（仓库不提交），
所以这条代价为零。

### 4.2 音频电平的在线估计

上游的特征提取器**每次调用都按该次调用的均值/标准差归一化**。离线时这没问题
（整段文件归一化一次）；流式时每个窗口各自归一化，特征尺度会逐块漂移，嘴型会抖。

本服务改为**慢速在线估计电平**（EMA，α=0.05，约 2 秒时间常数），所有窗口用同一个
尺度归一化。`tests/test_lite_audio.py` 钉住了这个行为。

### 4.3 帧栅格由采样数决定

和 MuseTalk 一样：每块的帧数由累计采样数算出（`total_samples * fps // sample_rate`），
100 ms 的 48 kHz 音频块 = 2 或 3 帧 25 fps，长期均值正好 25。用时钟而不是采样数，
10×0.1 这种浮点误差会让帧栅格慢慢漂移。

### 4.4 预取解码 + 批处理推理

实测（7800X3D，8 线程）：

| 环节 | 单帧 | 每块（3 帧） |
|---|---|---|
| FeatherHuBERT（500 ms 窗口） | — | 6–7 ms |
| 视觉 UNet | 17–20 ms | 52–60 ms |
| JPEG 解码 720×1280 | 8 ms | 25 ms |
| 裁剪 / 拼输入 / 贴回 | ~2 ms | 6 ms |

三处优化：

1. **批处理**：一块里的 2–3 帧合成一次前向（17.4 ms/帧 vs 20.1）。
2. **预取线程**：JPEG 解码放到独立线程，与模型前向重叠（解码 8 ms < 模型 17 ms，
   所以来得及）。
3. 帧以 JPEG 字节存在内存里（375 帧约 47 MB），按需解码并做 LRU —— 全解码常驻
   是 1 GB。

结果：**每块中位 76 ms、p95 87 ms**（优化前 98 ms），100 ms 预算的 76%，吞吐 35–36 fps。

### 4.5 帧数据放在形象包里

包里的帧就是训练时看到的帧（同一批 JPEG），所以"训练分辨率"和"服务分辨率"
不可能漂移。运行时按需解码，不常驻。

## 5. 包格式

```
<avatar_dir>/lite/
    frames/00000000.jpg   运动源帧（720×1280 画布，BGR 顺序的 JPEG）
    coords.json           每帧人脸框 {"bbox": [x1,y1,x2,y2], "size": [w,h]}
    visual.pth            专属视觉模型
    meta.json             版本 / 帧数 / 帧率 / 因果窗口标记
```

`pack.json` 里对应一个 `lite` 块。`AvatarStore.load_pack` 按**形象自己的
`drive_model`** 找块（`config.get(avatar.drive_model)`），所以包永远不会被
读它的引擎之外的引擎读走。

## 6. 与其它部分的关系

* `server/model_registry.py`：新增 `lite` 模型（`label_zh="轻量引擎 (FeatherTalk)"`），
  并给模型加了 `max_video_seconds` / `max_video_frames` 两个可选的源素材上限字段
  —— lite 的源是几分钟的训练视频，默认的 15 秒不够。
* `avatars.build_options`（新增列）：把 `train` / `checkpoint` / `epochs` 从创建表单
  持久化到形象行，重试制作时才能复现同样的构建。
* `POST /api/admin/avatars` 新增 `options` 表单字段（JSON）。
* 管理台：形象标签新增 `lite` 颜色；创建表单在选中 lite 时显示"训练专属模型"
  复选框与 checkpoint 路径输入框；视频时长提示按模型的上限显示。

## 7. 失败模式

| 情况 | 行为 |
|---|---|
| `vendor/FeatherTalk` 未 checkout | 引擎 `warmup` 抛 `EngineUnavailable`，提示跑 `scripts/setup_lite.sh` |
| 权重缺失 | 同上，提示 `scripts/fetch_lite_weights.sh`（22 MB） |
| 上传的是照片 | 流水线拒绝："轻量引擎需要一段人物口播视频" |
| 既不训练也没有 checkpoint | 拒绝："请勾选训练，或提供 checkpoint 路径" |
| 视频里人脸太少 | 超过 20% 的帧检测不到人脸即拒绝 |
| 没有音轨 | 训练路径拒绝（无法提取口型驱动所需的音频特征） |
| 训练失败 | 任务落库为 `failed`，错误写进 `avatar_jobs.error` |

## 8. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_lite_audio.py` | 重采样、帧栅格、**因果窗口**、起点补零、电平估计（无 torch 无 cv2） |
| `tests/test_lite_pack.py` | 包布局、`pack.json` 块、store 按模型取块（无 torch 无 cv2） |
| `tests/test_lite_causal_dataset.py` | **训练侧的因果填充与服务侧一致**（因果窗口是填充的不动点）（需 torch + vendor） |
| `tests/test_lite_geometry.py` | 人脸框、裁剪、模型输入的通道顺序、贴回（需 cv2） |
| `tests/test_lite_engine.py` | 帧数、帧格式、idle 路径、`describe()`（需 torch + cv2 + vendor） |
| `tests/test_lite_pipeline.py` | 拒绝照片 / 拒绝无 checkpoint、包的写入、进度上报（需 cv2） |

## 9. 唇形验证：用 MuseTalk 当老师

专属权重需要"这个人在说话"的素材。仓库里没有，但**有 MuseTalk**：同一张照片、
同一段语音，MuseTalk 驱动出来的就是这个人说话的视频 —— 那段视频就是训练素材。

```
照片 + 语音 ──MuseTalk(GPU)──► teacher.mp4（真人在说话，嘴型正确）
                                    │
                                    └─► FeatherTalk 训练（20 s / 15 epoch / 243 s）
                                              │
照片 + 语音 ──FeatherTalk(CPU)──► driven.mp4（嘴型跟着音频走）
```

2026-10-25 在本机跑通：

* 训练：loss 1.41 → 0.44 收敛，产出 5.46 M 参数的视觉模型（瘦身后的 checkpoint
  22.1 MB；训练产物带优化器状态，是 66 MB，服务端只需要模型）。
* 驱动：纯 CPU，**每块中位 72.6 ms / p95 81.9 ms**，37 fps。
* 画面：抽帧看嘴型随音节变化（`data/tmp/lite_train/contact_sheet.png`），视频在
  `data/tmp/lite_train/driven.mp4`。

> 这个"老师"办法不是 FeatherTalk 的用法，是**给没有素材的人一个起点**：先让
> 现有的重模型造出素材，再训轻模型。真正的生产素材当然是真人自己录的。

## 10. 已知取舍

* **CPU 实时性依赖机器**：实测基于 7800X3D。普通服务器 CPU 大约慢 1.5–2 倍，
  76 ms/块会变成 110–150 ms —— 刚好越界。所以"CPU 实时"目前成立的前提是
  一台较快的桌面级 CPU 或服务器 CPU。
* **训练需要 GPU**：`lite` 的推理是纯 CPU，但制作（训练专属权重）仍然要 GPU。
  纯 CPU 机器可以服务已有形象包，但不能新建。
* **FeatherHuBERT 只用中文语音训练**，且上游自己说训练不完整。英文语音的效果
  未验证。
* **音画同步**：因果窗口保证画面不落后于声音，但代价是模型看不到未来的协同发音
  信息，唇形精度略低于上游原生窗口。
* **画面边缘是硬的**：模型输出整块覆盖人脸框，框的边界没有羽化。这是上游的做法
  （`image[ymin:ymax, xmin:xmax] = face_crop`），没有改。
