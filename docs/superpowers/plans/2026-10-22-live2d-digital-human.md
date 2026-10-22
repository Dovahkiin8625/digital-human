# Live2D Digital Human Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一个 digital-human 服务内增加 Live2D 数字人形态，与现有真人视频数字人并存，交付一个网页客户端和一个 Electron 桌面宠物，两者共用同一份前端 bundle。

**Architecture:** 服务端不渲染 Live2D 画面，只提供模型托管与业务面 LLM/TTS 端点；口型完全在客户端由正在播放的 `AudioBuffer` 离线预算出的包络驱动，因此嘴与声音天然零漂移。渲染层是一个独立的 `live2d-ui` Vite app，构建产物既挂到 FastAPI 的 `/live2d`，也打进 Electron。

**Tech Stack:** TypeScript / React 18 / Vite 5 / vitest；Cubism SDK for Web；Electron。服务端沿用 FastAPI + SQLAlchemy + SQLite + pytest（torch-free）。

**Spec:** `docs/superpowers/specs/2026-10-22-live2d-digital-human-design.md` —— 计划随 spec 一起流转，执行者两个都要读。

---

## Global Constraints

- **不改任何现有渲染路径。** MuseTalk / LivePortrait / Wan 2.2 的引擎、制作流水线、WebRTC 出口、信令协议一行不动。
- **不引入 ASR。** v1 仅文字输入。
- **不给 Live2D 开 session 类型。** 文字输入下客户端自足（spec §3.2）。
- **`/api/v1/*` 目前完全无鉴权**，这是既有状态，不要顺手"修复"现有端点。新端点单独用 `DH_V1_API_TOKEN`（spec §4.3）。
- **服务端只套 WAV 头，重采样放浏览器。** 沿用现有做法，服务端不得引入音频编解码依赖。
- **参数名与取值范围必须在运行时从模型读取**，禁止硬编码 `0..1`（spec §5.1）。
- **任何一环出错，idle 动画必须继续运行**（spec §7）。
- **测试一律用 `client` fixture**（`tests/conftest.py`），它已把 `DH_DATA_DIR` 与 `DH_DATABASE_URL` 指向 `tmp_path`。**不得触碰 `data/digital-human.db`** —— 本地管理台的 demo key 绑在真库上。
- 默认 `pytest` 必须保持 torch-free 且十秒内跑完（`pyproject.toml` 的 `addopts = "-m 'not gpu and not musetalk'"`）。

## Review Focus

以下五条是 spec 暗示、但没有任何任务的测试覆盖、最可能真的伤到人的行为。每条都已在对应任务中补了测试。

1. **`[think]` 这类情绪标记泄漏进 TTS 文本 → 宠物把标记当台词念出来。** 标记必须从送进 TTS 的文本里剥干净，而不是只在 UI 上藏起来。（Task 8）
2. **模型没有 `ParamMouthOpenY` → 加载时抛异常、宠物直接消失。** 应当降级为嘴不动、表情与生命感照常。（Task 4）
3. **模型参数范围是 `-1..1` 或 `0..30` → 嘴钉死在最大值或开合反相。** 范围必须从模型规格读出后做线性映射。（Task 4）
4. **Electron 窗口失焦 → 动画冻结，宠物变成一张静止图片。** 这是桌面宠物最致命的一条：`backgroundThrottling` 默认为 true，而宠物永远失焦。（Task 11）
5. **TTS 返回近静音音频 → 峰值归一化把底噪放大成永久半张嘴。** 归一化前必须先过静音门限。（Task 2）

---

## File Structure

**新增 —— 前端 `live2d-ui/`（独立 Vite app，不混入 admin-ui）**

| 文件 | 职责 |
|---|---|
| `live2d-ui/vite.config.ts` | `base: "/live2d/"`，dev 代理 `/api`（含 `ws: true`） |
| `live2d-ui/package.json` | 含 `vitest`；admin-ui **尚未**接入 vitest，此处是全新 app，自带即可 |
| `live2d-ui/src/audio/analyze.ts` | `analyzeSpeech()`：PCM → 每帧口型包络与口型形状。纯函数，无 DOM 依赖 |
| `live2d-ui/src/render/params.ts` | `resolveMouthRange()`：从模型规格解析口型参数的真实取值范围 |
| `live2d-ui/src/render/idle.ts` | `idlePose()`：时刻 → 呼吸/眨眼/扫视/头部微动。纯函数、可确定性重放 |
| `live2d-ui/src/render/expressions.ts` | `expressionIndex()`：情绪标记 → 表情索引，未知降级 neutral |
| `live2d-ui/src/render/frame.ts` | `applyFrame()`：按 Cubism 要求的顺序编排每一帧 |
| `live2d-ui/src/api/client.ts` | `streamChat()` / `synthesize()`：带 `X-V1-Token` 的 SSE 与 TTS 调用 |
| `live2d-ui/src/api/sentences.ts` | `splitSentences()` / `stripEmotionTags()`：分句与标记剥离 |
| `live2d-ui/src/App.tsx` | 装配：canvas + 会话 + 句子队列 |
| `live2d-ui/electron/main.cjs` | Electron 主进程：透明置顶窗、穿透、托盘 |
| `live2d-ui/electron/preload.cjs` | contextBridge：仅暴露配置读写 |

**新增 —— 服务端**

| 文件 | 职责 |
|---|---|
| `server/core/auth.py` | `require_v1_token`：业务面端点的 token 依赖 |
| `server/live2d/registry.py` | `validate_model_dir()`：注册即校验模型完整性 |
| `server/api/client.py` | 业务面路由：`/api/v1/chat/stream`、`/api/v1/tts` |

**修改**

| 文件 | 改动 |
|---|---|
| `server/db/models.py` | `Avatar` 加 `kind` 列 |
| `server/db/session.py` | `_SCHEMA_MIGRATIONS` 加一行；`_drop_abandoned_tenant_schema` 的 `keep` 集合加 `kind` |
| `server/config.py` | 加 `v1_api_token` |
| `server/api/avatars.py` | 列表端点加 `kind` 过滤 |
| `server/main.py` | 挂载 `/live2d` |
| `README.md` | 修正过时的 `X-API-Key` 描述，补新端点 |

> ⚠️ `server/db/session.py` 里的 `_drop_abandoned_tenant_schema` 有一个 `keep` 键集合，是旧库重建 `avatars` 表时白名单的列。**加了 `kind` 却忘了把它加进 `keep`，旧库一旦触发重建就会静默丢列。** 这是本计划最容易踩的坑。

---

## Phase 0 — 可行性验证（丢弃性）

### Task 1: Cubism SDK 可行性验证

这是唯一可能否证整个方案的外部未知（spec §9）。**在它通过之前不要写任何生产代码。** 产出是一个答案，不是要保留的代码。

**Files:**
- Create（全部在 gitignore 的 `spike/` 下，**不提交**）: `spike/live2d-sdk/`

**Interfaces:**
- Consumes: 无
- Produces: 一个书面结论 —— SDK 能否安装、模型能否装载、参数规格能否读出。这个结论决定 Phase 1 是否继续。

- [ ] **Step 1: 确认 Cubism SDK 可获取**

访问 Live2D 开发者官网，下载 Cubism SDK for Web，确认拿到的是带 `.framework.js` 的 SDK 包，并完成 License Agreement 的账号注册。

- [ ] **Step 2: 记录 SDK 的实际包名与安装方式**

把 SDK 解压，记录 `Core` 目录下的 framework 文件名与最小可用示例所需的全部文件。SDK 不走公共 npm registry，安装方式取决于你拿到的包的形态（通常是拷贝 `Core` 到本地并手写 import 路径）。

- [ ] **Step 3: 最小 Electron 渲染 demo**

在 `spike/live2d-sdk/` 下建一个最小 Electron 应用：透明无边框窗口 + 一个 `<canvas>` + 从磁盘加载一个真实 Live2D 模型。不需要任何服务端。

- [ ] **Step 4: 验证参数规格可读出**

在渲染循环里打印模型的 `ParamMouthOpenY`、`ParamAngleZ`、`ParamEyeLOpen`、`ParamEyeBallX` 的实际 min/max。**这是整个 Phase 1 的前提** —— 如果读不出范围，口型映射就只能写死，而写死一定错（spec §5.1）。

- [ ] **Step 5: 验证失焦时动画继续**

把窗口失焦（点别处），确认模型仍在动。若冻结，立刻验证 `backgroundThrottling: false` 能解决 —— 这条会成为 Task 11 的实现依据。

- [ ] **Step 6: 写下结论并做决策**

在对话中明确回答三件事：**SDK 可用 / 不可用**；参数规格**可读 / 不可读**；失焦问题**已定位到 `backgroundThrottling` / 另有原因**。

> 若 SDK 不可获取（许可条款不接受），**停下来回到 brainstorming**，与需求方讨论改走 `pixi-live2d-display` 或 VRM 路线。不要自行替换技术栈。

---

## Phase 1 — 渲染内核（live2d-ui）

### Task 2: 搭建 live2d-ui 并实现语音分析

**Files:**
- Create: `live2d-ui/package.json`, `live2d-ui/vite.config.ts`, `live2d-ui/tsconfig.json`, `live2d-ui/index.html`
- Create: `live2d-ui/src/audio/analyze.ts`, `live2d-ui/src/audio/analyze.test.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  ```ts
  export interface PcmSource {
    readonly numberOfChannels: number
    readonly sampleRate: number
    readonly length: number
    getChannelData(channel: number): Float32Array
  }
  export interface MouthAnalysis {
    open: Float32Array   // 每帧张口度 0..1
    form: Float32Array   // 每帧口型形状 0..1
  }
  export interface AnalyzeOptions { pointsPerSecond?: number }
  export function analyzeSpeech(src: PcmSource, opts?: AnalyzeOptions): MouthAnalysis
  ```
  `AudioBuffer` 在结构上满足 `PcmSource`，因此可以直接传入真实的解码结果，但测试无需 DOM。

- [ ] **Step 1: 脚手架 + vitest 配置**

`package.json` 依赖与 `admin-ui` 对齐（react 18 / vite 5 / typescript 5），**外加 `vitest`**。`vite.config.ts` 必须设 `base: "/live2d/"`（构建产物由 FastAPI 挂到该前缀，不设则 asset URL 404），并照抄 `admin-ui/vite.config.ts` 的 `server.proxy` —— 包括那个 `ws: true`，它不是可选的。

`analyze.test.ts` 顶部放一个 `pcm()` 工厂，供后续所有前端测试复用：

```ts
export function pcm(channels: Float32Array[], sampleRate: number): PcmSource {
  return {
    numberOfChannels: channels.length,
    sampleRate,
    length: channels[0].length,
    getChannelData: (c: number) => channels[c],
  }
}
```

- [ ] **Step 2: 写失败的测试**

`analyze.test.ts`：

```ts
import { describe, expect, it } from "vitest"
import { analyzeSpeech, type PcmSource } from "./analyze"

function pcm(channels: Float32Array[], sampleRate = 48000): PcmSource {
  return {
    numberOfChannels: channels.length,
    sampleRate,
    length: channels[0].length,
    getChannelData: (c: number) => channels[c],
  }
}

function tone(seconds: number, sampleRate = 48000, amp = 0.5, hz = 220): Float32Array {
  const n = Math.round(seconds * sampleRate)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * hz * i) / sampleRate)
  return out
}

describe("analyzeSpeech", () => {
  // Review Focus #5：近静音音频不得被归一化成永久半张嘴。
  it("returns all zeros for silence", () => {
    const r = analyzeSpeech(pcm([tone(0.5, 48000, 0)]))
    expect(r.open.every((v) => v === 0)).toBe(true)
    expect(r.form.every((v) => v === 0)).toBe(true)
  })

  it("returns all zeros when only near-silence is present", () => {
    const n = 48000
    const noise = new Float32Array(n)
    for (let i = 0; i < n; i++) noise[i] = (i % 7) * 1e-4   // 远低于静音门限
    const r = analyzeSpeech(pcm([noise]))
    expect(Math.max(...r.open)).toBe(0)
  })

  it("peaks at 1.0 regardless of input amplitude", () => {
    const quiet = analyzeSpeech(pcm([tone(0.5, 48000, 0.05)]))
    const loud = analyzeSpeech(pcm([tone(0.5, 48000, 0.9)]))
    expect(Math.max(...quiet.open)).toBeCloseTo(1, 2)
    expect(Math.max(...loud.open)).toBeCloseTo(1, 2)
  })

  it("emits one value per pointsPerSecond frame", () => {
    const r = analyzeSpeech(pcm([tone(1.0, 48000)]), { pointsPerSecond: 60 })
    expect(r.open.length).toBe(60)
  })

  // 非对称平滑：开得快、闭得慢。
  it("closes more slowly than it opens", () => {
    const half = 24000
    const buf = new Float32Array(48000)
    buf.fill(0.5, 0, half)          // 前 0.5s 有声
    const r = analyzeSpeech(pcm([buf]), { pointsPerSecond: 60 })
    const peakIdx = r.open.indexOf(Math.max(...r.open))
    const framesToHalf = r.open.findIndex((v, i) => i > peakIdx && v < 0.5)
    expect(framesToHalf - peakIdx).toBeGreaterThan(1)   // 衰减是多帧的，不是阶跃
  })

  it("averages across channels", () => {
    const a = tone(0.5)
    const b = new Float32Array(a.length)                // 右声道静音
    const both = analyzeSpeech(pcm([a, b]))
    const leftOnly = analyzeSpeech(pcm([a]))
    expect(Math.max(...both.open)).toBeCloseTo(Math.max(...leftOnly.open), 2)
  })
})
```

- [ ] **Step 3: 运行，确认失败**

```
cd live2d-ui && npx vitest run src/audio/analyze.test.ts
```
预期：FAIL，报 `Cannot find module './analyze'`。

- [ ] **Step 4: 实现**

`live2d-ui/src/audio/analyze.ts`：

```ts
export interface PcmSource {
  readonly numberOfChannels: number
  readonly sampleRate: number
  readonly length: number
  getChannelData(channel: number): Float32Array
}

export interface MouthAnalysis {
  open: Float32Array
  form: Float32Array
}

export interface AnalyzeOptions {
  pointsPerSecond?: number
}

const DEFAULT_POINTS_PER_SECOND = 60

/** Below this the clip is treated as silence, not as a quiet utterance. */
const SILENCE_RMS = 0.005
const ATTACK = 0.6
const RELEASE = 0.18
/** Below this mouth openness the zero-crossing count is noise, not a shape. */
const FORM_FLOOR = 0.1

export function analyzeSpeech(
  src: PcmSource,
  { pointsPerSecond = DEFAULT_POINTS_PER_SECOND }: AnalyzeOptions = {},
): MouthAnalysis {
  const frames = Math.max(1, Math.ceil((src.length / src.sampleRate) * pointsPerSecond))
  const perFrame = src.length / frames
  const channels: Float32Array[] = []
  for (let c = 0; c < src.numberOfChannels; c++) channels.push(src.getChannelData(c))

  const rms = new Float32Array(frames)
  const zcr = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    const start = Math.floor(f * perFrame)
    const end = Math.min(src.length, Math.floor((f + 1) * perFrame))
    let sumSq = 0
    let crossings = 0
    let prev = 0
    for (const data of channels) {
      for (let i = start; i < end; i++) {
        const s = data[i]
        sumSq += s * s
        if (i > start && (s >= 0) !== (prev >= 0)) crossings++
        prev = s
      }
    }
    const count = (end - start) * channels.length
    rms[f] = count > 0 ? Math.sqrt(sumSq / count) : 0
    zcr[f] = count > 0 ? crossings / count : 0
  }

  // Gate on absolute level BEFORE normalizing: peak-normalizing a clip that
  // is only noise would stretch that noise across the full 0..1 range and
  // leave the mouth permanently half open.
  let peak = 0
  for (const v of rms) if (v > peak) peak = v
  const gain = peak > SILENCE_RMS ? 1 / peak : 0

  // Asymmetric one-pole. A symmetric filter reads as a pulsing sine; the
  // fast-open / slow-close shape is what reads as speech.
  const open = new Float32Array(frames)
  let env = 0
  for (let f = 0; f < frames; f++) {
    const target = Math.min(1, rms[f] * gain)
    env += (target - env) * (target > env ? ATTACK : RELEASE)
    open[f] = env
  }

  let zPeak = 0
  for (const v of zcr) if (v > zPeak) zPeak = v
  const zGain = zPeak > 0 ? 1 / zPeak : 0
  const form = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    form[f] = open[f] > FORM_FLOOR ? Math.min(1, zcr[f] * zGain) : 0
  }

  return { open, form }
}
```

- [ ] **Step 5: 运行，确认通过**

```
npx vitest run src/audio/analyze.test.ts
```
预期：5 个测试全 PASS。

- [ ] **Step 6: 提交**

```bash
git add live2d-ui/
git commit -m "feat(live2d-ui): scaffold the client app and the mouth-envelope analysis"
```

---

### Task 3: idle 生命感层

**Files:**
- Create: `live2d-ui/src/render/idle.ts`, `live2d-ui/src/render/idle.test.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  ```ts
  export interface IdlePose {
    angleX: number; angleY: number; angleZ: number
    eyeOpen: number
    eyeBallX: number; eyeBallY: number
  }
  export interface IdleOptions { breathScale?: number; seed?: number }
  export function idlePose(tMs: number, opts?: IdleOptions): IdlePose
  ```

- [ ] **Step 1: 写失败的测试**

```ts
import { describe, expect, it } from "vitest"
import { idlePose } from "./idle"

describe("idlePose", () => {
  it("is a pure function of time", () => {
    expect(idlePose(12345, { seed: 7 })).toEqual(idlePose(12345, { seed: 7 }))
  })

  // 三层呼吸周期互质，循环才不会在某个可察觉的周期上对齐。
  it("breathes continuously with no reset", () => {
    const a = idlePose(10_000)
    const b = idlePose(10_016)
    expect(Math.abs(a.angleZ - b.angleZ)).toBeLessThan(0.1)
  })

  it("scales breathing by breathScale without shifting phase", () => {
    const t = 900
    const full = idlePose(t, { breathScale: 1 })
    const half = idlePose(t, { breathScale: 0.5 })
    expect(half.angleZ).toBeCloseTo(full.angleZ * 0.5, 6)
  })

  it("keeps the eyes open most of the time", () => {
    let open = 0
    for (let t = 0; t < 60_000; t += 40) if (idlePose(t, { seed: 3 }).eyeOpen > 0.9) open++
    expect(open / 1500).toBeGreaterThan(0.8)
  })

  it("closes the eyes sometimes", () => {
    let closed = 0
    for (let t = 0; t < 60_000; t += 40) if (idlePose(t, { seed: 3 }).eyeOpen < 0.3) closed++
    expect(closed).toBeGreaterThan(0)
  })

  it("varies blink timing across seeds", () => {
    const closedAt = (seed: number) => {
      for (let t = 0; t < 20_000; t += 40) if (idlePose(t, { seed }).eyeOpen < 0.3) return t
      return -1
    }
    expect(closedAt(1)).not.toBe(closedAt(2))
  })
})
```

- [ ] **Step 2: 运行，确认失败**

`npx vitest run src/render/idle.test.ts` → FAIL，`Cannot find module './idle'`。

- [ ] **Step 3: 实现**

`live2d-ui/src/render/idle.ts`：

```ts
export interface IdlePose {
  angleX: number
  angleY: number
  angleZ: number
  eyeOpen: number
  eyeBallX: number
  eyeBallY: number
}

export interface IdleOptions {
  /** 0..1 multiplier on breathing/sway amplitude. Lowered while speaking. */
  breathScale?: number
  /** Seeds the blink and saccade schedule so it is replayable in tests. */
  seed?: number
}

const BREATH_MS = 4000
const BLINK_SLOT_MS = 4000
const BLINK_MS = 150
const SACCADE_SLOT_MS = 1200
const EYE_MIN = 0.05

/** Deterministic 0..1 hash. Lets every timing decision be a pure function
 *  of (seed, bucket) instead of carrying mutable PRNG state through the
 *  render loop -- which is what makes the whole layer replayable in a test. */
function hash01(seed: number, i: number): number {
  let t = Math.imul(seed ^ i, 0x9e3779b1) >>> 0
  t = Math.imul(t ^ (t >>> 16), 0x85ebca6b) >>> 0
  t = Math.imul(t ^ (t >>> 13), 0xc2b2ae35) >>> 0
  return ((t ^ (t >>> 16)) >>> 0) / 4294967296
}

export function idlePose(tMs: number, opts: IdleOptions = {}): IdlePose {
  const { breathScale = 1, seed = 1 } = opts

  // Three mutually incommensurable periods so the composite never visibly
  // repeats. Phase is a function of t alone -- never reset per sentence,
  // which would put a jump at every utterance boundary.
  const s = (2 * Math.PI * tMs) / BREATH_MS
  const angleZ = Math.sin(s) * 2.5 * breathScale
  const angleX = (Math.sin(s / 1.7 + 0.4) * 1.2 + Math.sin(s / 4.1) * 0.6) * breathScale
  const angleY = (Math.sin(s / 2.3 + 1.1) * 1.0 + Math.sin(s / 3.3) * 0.5) * breathScale

  const blinkSlot = Math.floor(tMs / BLINK_SLOT_MS)
  const inSlot = tMs - blinkSlot * BLINK_SLOT_MS
  let eyeOpen = 1
  if (hash01(seed, blinkSlot) < 0.45) {
    const at = hash01(seed, blinkSlot + 1) * (BLINK_SLOT_MS - BLINK_MS * 3)
    const d = inSlot - at
    if (d >= 0 && d < BLINK_MS) {
      eyeOpen = EYE_MIN + (1 - EYE_MIN) * (1 - Math.sin((Math.PI * d) / BLINK_MS))
    }
    // Occasional double blink, gated by its own hash so it is not universal.
    if (hash01(seed, blinkSlot + 2) < 0.25) {
      const d2 = inSlot - at - 380
      if (d2 >= 0 && d2 < BLINK_MS) {
        eyeOpen = Math.min(eyeOpen, EYE_MIN + (1 - EYE_MIN) * (1 - Math.sin((Math.PI * d2) / BLINK_MS)))
      }
    }
  }

  const sacSlot = Math.floor(tMs / SACCADE_SLOT_MS)
  const eyeBallX = (hash01(seed, sacSlot + 1000) - 0.5) * 0.6
  const eyeBallY = (hash01(seed, sacSlot + 2000) - 0.5) * 0.4

  return { angleX, angleY, angleZ, eyeOpen, eyeBallX, eyeBallY }
}
```

- [ ] **Step 4: 运行，确认通过**

`npx vitest run src/render/idle.test.ts` → 6 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
git add live2d-ui/src/render/
git commit -m "feat(live2d-ui): add the deterministic idle-life layer"
```

---

### Task 4: 参数解析与帧序编排

**Files:**
- Create: `live2d-ui/src/render/params.ts`, `live2d-ui/src/render/params.test.ts`
- Create: `live2d-ui/src/render/frame.ts`, `live2d-ui/src/render/frame.test.ts`
- Create: `live2d-ui/src/render/expressions.ts`, `live2d-ui/src/render/expressions.test.ts`

**Interfaces:**
- Consumes: `IdlePose` from Task 3
- Produces:
  ```ts
  // params.ts
  export interface ParameterSpec { min: number; max: number; default: number }
  export interface ParamIndex { getParameterIndex(name: string): number; getParameterMinimumValue(i: number): number; getParameterMaximumValue(i: number): number }
  export function resolveMouthRange(model: ParamIndex): { min: number; max: number } | null
  export function mapToRange(value01: number, range: { min: number; max: number }): number

  // expressions.ts
  export function expressionIndex(tags: string[], map: Record<string, number>, fallback = 0): number

  // frame.ts
  export interface FrameInputs {
    model: {
      setParameterValueByIndex(i: number, v: number): void
      getParameterIndex(name: string): number
      getParameterMinimumValue(i: number): number
      getParameterMaximumValue(i: number): number
    }
    expression: { update(): void }
    motion: { update(): void }
    physics: { update(): void }
    model3: { update(): void }
    expressionIndex: number
    open: number
    form: number
    idle: IdlePose
    breathScale: number
  }
  export function applyFrame(f: FrameInputs): void
  ```

- [ ] **Step 1: 写失败的测试**

`params.test.ts`：

```ts
import { describe, expect, it } from "vitest"
import { mapToRange, resolveMouthRange } from "./params"

function fakeModel(specs: Record<string, { min: number; max: number }>) {
  const names = Object.keys(specs)
  return {
    getParameterIndex: (n: string) => names.indexOf(n),
    getParameterMinimumValue: (i: number) => specs[names[i]].min,
    getParameterMaximumValue: (i: number) => specs[names[i]].max,
  }
}

describe("resolveMouthRange", () => {
  // Review Focus #3：范围不是 0..1 的模型不得被钉死或反相。
  it.each([
    [0, 1],
    [-1, 1],
    [0, 30],
    [-30, 30],
  ])("reads a %i..%i range off the model", (min, max) => {
    const r = resolveMouthRange(fakeModel({ ParamMouthOpenY: { min, max } }))
    expect(r).toEqual({ min, max })
  })

  // Review Focus #2：没有口型参数的模型必须降级，不能抛。
  it("returns null when the model has no mouth parameter", () => {
    expect(resolveMouthRange(fakeModel({ ParamAngleZ: { min: -30, max: 30 } }))).toBeNull()
  })
})

describe("mapToRange", () => {
  it("maps 0 and 1 onto the range endpoints", () => {
    expect(mapToRange(0, { min: 0, max: 30 })).toBe(0)
    expect(mapToRange(1, { min: 0, max: 30 })).toBe(30)
  })
  it("maps the midpoint to the range midpoint", () => {
    expect(mapToRange(0.5, { min: -1, max: 1 })).toBe(0)
  })
  it("clamps out-of-range input", () => {
    expect(mapToRange(1.4, { min: 0, max: 1 })).toBe(1)
    expect(mapToRange(-0.2, { min: 0, max: 1 })).toBe(0)
  })
})
```

`frame.test.ts` —— 用一个记录调用顺序的假模型，把 Cubism 的顺序要求钉死：

```ts
import { describe, expect, it } from "vitest"
import { applyFrame } from "./frame"

function recorder() {
  const calls: string[] = []
  return {
    calls,
    model: {
      setParameterValueByIndex: (_i: number, _v: number) => calls.push("setParam"),
      getParameterIndex: () => 0,
      getParameterMinimumValue: () => 0,
      getParameterMaximumValue: () => 1,
    },
    expression: { update: () => calls.push("expression") },
    motion: { update: () => calls.push("motion") },
    physics: { update: () => calls.push("physics") },
    model3: { update: () => calls.push("model") },
  }
}

describe("applyFrame", () => {
  it("writes our params after Cubism's managers so we win the conflict", () => {
    const r = recorder()
    applyFrame({
      ...r,
      expressionIndex: 0, open: 0.5, form: 0.2,
      idle: { angleX: 0, angleY: 0, angleZ: 1, eyeOpen: 1, eyeBallX: 0, eyeBallY: 0 },
      breathScale: 1,
    })
    const firstParam = r.calls.indexOf("setParam")
    expect(firstParam).toBeGreaterThan(r.calls.indexOf("expression"))
    expect(firstParam).toBeGreaterThan(r.calls.indexOf("motion"))
  })

  it("runs physics before the pose bake", () => {
    const r = recorder()
    applyFrame({
      ...r,
      expressionIndex: 0, open: 0, form: 0,
      idle: { angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 1, eyeBallX: 0, eyeBallY: 0 },
      breathScale: 1,
    })
    expect(r.calls.indexOf("physics")).toBeLessThan(r.calls.indexOf("model"))
  })

  it("leaves the model untouched when there is no mouth parameter", () => {
    const r = recorder()
    r.model.getParameterIndex = () => -1
    applyFrame({
      ...r,
      expressionIndex: 0, open: 1, form: 1,
      idle: { angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 1, eyeBallX: 0, eyeBallY: 0 },
      breathScale: 1,
    })
    // Only the idle writes survive; nothing threw.
    expect(r.calls.filter((c) => c === "setParam").length).toBeGreaterThan(0)
    expect(r.calls).toContain("model")
  })
})
```

`expressions.test.ts`：

```ts
import { describe, expect, it } from "vitest"
import { expressionIndex } from "./expressions"

describe("expressionIndex", () => {
  it("maps a known tag", () => {
    expect(expressionIndex(["smile"], { smile: 3 })).toBe(3)
  })
  it("uses the last tag when several arrive in one sentence", () => {
    expect(expressionIndex(["smile", "think"], { smile: 3, think: 5 })).toBe(5)
  })
  it("falls back to neutral for an unknown tag", () => {
    expect(expressionIndex(["wink"], { smile: 3 }, 0)).toBe(0)
  })
  it("falls back to neutral for no tag", () => {
    expect(expressionIndex([], { smile: 3 }, 0)).toBe(0)
  })
})
```

- [ ] **Step 2: 运行，确认三个文件都失败**

```
npx vitest run src/render/
```
预期：三个 `Cannot find module`。

- [ ] **Step 3: 实现 `params.ts`**

```ts
export interface ParamIndex {
  getParameterIndex(name: string): number
  getParameterMinimumValue(i: number): number
  getParameterMaximumValue(i: number): number
}

export function resolveMouthRange(model: ParamIndex): { min: number; max: number } | null {
  const i = model.getParameterIndex("ParamMouthOpenY")
  // Cubism returns -1 for a name the model does not define. Some rigs ship
  // a differently-spelled or absent mouth parameter entirely, and those
  // models must degrade to "mouth does not move", not throw.
  if (i < 0) return null
  return { min: model.getParameterMinimumValue(i), max: model.getParameterMaximumValue(i) }
}

export function mapToRange(value01: number, range: { min: number; max: number }): number {
  const v = Math.min(1, Math.max(0, value01))
  return range.min + v * (range.max - range.min)
}
```

- [ ] **Step 4: 实现 `expressions.ts`**

```ts
/** Resolve the emotion tags seen in the current sentence to an expression
 *  index. Unknown tags resolve to the fallback rather than throwing -- a
 *  model the user swapped in at runtime must not be able to blank the pet. */
export function expressionIndex(
  tags: string[],
  map: Record<string, number>,
  fallback = 0,
): number {
  for (let i = tags.length - 1; i >= 0; i--) {
    const hit = map[tags[i]]
    if (hit !== undefined) return hit
  }
  return fallback
}
```

- [ ] **Step 5: 实现 `frame.ts`**

```ts
import { mapToRange, resolveMouthRange } from "./params"
import type { IdlePose } from "./idle"

export interface FrameInputs {
  model: {
    setParameterValueByIndex(i: number, v: number): void
    getParameterIndex(name: string): number
    getParameterMinimumValue(i: number): number
    getParameterMaximumValue(i: number): number
  }
  expression: { update(): void }
  motion: { update(): void }
  physics: { update(): void }
  model3: { update(): void }
  expressionIndex: number
  open: number
  form: number
  idle: IdlePose
  breathScale: number
}

const MOUTH_FORM_FALLBACK = { min: -1, max: 1 }

export function applyFrame(f: FrameInputs): void {
  // 1-2. Cubism lays down expression and motion first.
  f.expression.update()
  f.motion.update()

  // 3. We write afterwards so our values win any parameter an expression
  // also touches. Writing before the managers run is the single most
  // common cause of mouth-vs-expression flicker.
  const mouthRange = resolveMouthRange(f.model)
  if (mouthRange) {
    const i = f.model.getParameterIndex("ParamMouthOpenY")
    f.model.setParameterValueByIndex(i, mapToRange(f.open, mouthRange))
  }
  const formIdx = f.model.getParameterIndex("ParamMouthForm")
  if (formIdx >= 0) {
    f.model.setParameterValueByIndex(formIdx, mapToRange(f.form, MOUTH_FORM_FALLBACK))
  }

  const set = (name: string, v: number) => {
    const i = f.model.getParameterIndex(name)
    if (i >= 0) f.model.setParameterValueByIndex(i, v)
  }
  set("ParamAngleX", f.idle.angleX * f.breathScale)
  set("ParamAngleY", f.idle.angleY * f.breathScale)
  set("ParamAngleZ", f.idle.angleZ * f.breathScale)
  set("ParamEyeLOpen", f.idle.eyeOpen)
  set("ParamEyeBallX", f.idle.eyeBallX)
  set("ParamEyeBallY", f.idle.eyeBallY)

  // 4-5. Physics derives hair/cloth from the head params we just wrote,
  // then the pose is baked into vertices.
  f.physics.update()
  f.model3.update()
}
```

- [ ] **Step 6: 运行，确认全部通过**

```
npx vitest run src/render/
```
预期：`params` 7 个、`frame` 3 个、`expressions` 4 个，全 PASS。

- [ ] **Step 7: 提交**

```bash
git add live2d-ui/src/render/
git commit -m "feat(live2d-ui): resolve real parameter ranges and enforce Cubism frame order"
```

---

### Task 5: 调试页（接真实模型）

本任务**不做单元测试** —— WebGL 不可单测，验证方式是真人看。目的是在写服务端之前确认内核真的能驱动一个真实模型。

**Files:**
- Create: `live2d-ui/src/main.tsx`, `live2d-ui/src/DebugStage.tsx`, `live2d-ui/src/App.tsx`, `live2d-ui/src/styles.css`

**Interfaces:**
- Consumes: `analyzeSpeech`、`idlePose`、`applyFrame`（Tasks 2–4）
- Produces: 一个能在浏览器里转起来、能接麦克风让嘴动的页面

- [ ] **Step 1: 实现渲染循环与模型装载**

用 Task 1 spike 里确认可用的那套 SDK 装载代码。循环固定为：装载模型 → 关掉 Cubism 的自动 breath/eyeblink/lipsync → `requestAnimationFrame` 里调 `applyFrame`。

> 关自动动作是铁律。Cubism 内置的 breath 与我们的 idle 层同时写 `ParamAngleZ`，两边打架必然抖动。

- [ ] **Step 2: 接一个音频输入做口型验证**

页面放一个「加载音频文件」按钮，读 WAV → `decodeAudioData` → `analyzeSpeech` → 播放时按 `currentTime - startTime` 查表喂 `open` / `form`。

- [ ] **Step 3: 人工验证清单**

- [ ] 模型能转，idle 在动
- [ ] 加载音频后嘴随之开合，**音画同步**（嘴动与出声同时开始、同时结束）
- [ ] 拖动窗口 / 切到别的标签页再回来，**动画没有冻结**
- [ ] 换几个参数范围不同的模型，嘴都没有钉死或反相
- [ ] 控制台无报错

- [ ] **Step 4: 提交**

```bash
git add live2d-ui/src/
git commit -m "feat(live2d-ui): drive a real model from the debug stage"
```

---

## Phase 2 — 服务端

### Task 6: `kind` 列与迁移

**Files:**
- Modify: `server/db/models.py:20-36`（`Avatar` 加 `kind`）
- Modify: `server/db/session.py`（`_SCHEMA_MIGRATIONS` 加一行；`_drop_abandoned_tenant_schema` 的 `keep` 加 `"kind"`）
- Test: `tests/test_live2d_kind_migration.py`

**Interfaces:**
- Consumes: 无
- Produces: `Avatar.kind: str`，取值 `"photoreal"`（默认）或 `"live2d"`；设置项 `settings.v1_api_token: str`（Task 7 用）

- [ ] **Step 1: 写失败的测试**

```python
"""`Avatar.kind` migration: additive, idempotent, and it must survive the
tenant-schema rebuild. That rebuild whitelists columns by name in
``_drop_abandoned_tenant_schema``; a column missing from that whitelist is
silently dropped the first time an old database triggers it."""

from sqlalchemy import inspect, text

from server.db.session import get_engine, init_db


def test_kind_defaults_to_photoreal(client):
    resp = client.post(
        "/api/v1/avatars",
        files={"file": ("a.png", b"x", "image/png")},
        data={"name": "a"},
    )
    assert resp.status_code in (200, 202)
    avatar_id = resp.json()["avatar_id"]
    assert client.get(f"/api/v1/avatars/{avatar_id}").json()["kind"] == "photoreal"


def test_migration_is_idempotent():
    engine = get_engine()
    init_db(engine)
    init_db(engine)          # second run must be a no-op, not an error
    cols = {c["name"] for c in inspect(engine).get_columns("avatars")}
    assert "kind" in cols


def test_tenant_rebuild_keep_list_includes_kind():
    """Guards the whitelist itself: the rebuild drops any column not named
    in ``keep``. Pin it so adding a column without updating the list fails
    here rather than in production."""
    import inspect as pyinspect

    from server.db import session as session_mod

    src = pyinspect.getsource(session_mod._drop_abandoned_tenant_schema)
    assert '"kind"' in src, "avatars rebuild whitelist is missing 'kind'"
```

- [ ] **Step 2: 运行，确认失败**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_kind_migration.py -v
```
预期：`test_kind_defaults_to_photoreal` FAIL（响应里没有 `kind`）。

- [ ] **Step 3: 改模型**

`server/db/models.py` 的 `Avatar` 里加：

```python
    #: Which renderer this avatar is for. ``photoreal`` rows go through the
    #: MuseTalk / LivePortrait / Wan pack pipeline; ``live2d`` rows are a
    #: Cubism project served to the client, which renders it itself. Adding
    #: a third renderer means a third value here, not a second table.
    kind: Mapped[str] = mapped_column(String(16), default="photoreal", nullable=False)
```

- [ ] **Step 4: 加迁移**

`server/db/session.py` 的 `_SCHEMA_MIGRATIONS` 加：

```python
    (
        "avatars",
        "kind",
        "ALTER TABLE avatars ADD COLUMN kind VARCHAR(16) NOT NULL DEFAULT 'photoreal'",
    ),
```

同时把 `_drop_abandoned_tenant_schema` 里 avatars 的 `keep` 元组补上 `"kind"`：

```python
        keep = {name: avatars_cols[name] for name in (
            "id", "name", "status", "source_image_path", "pack_path",
            "config_json", "tunable_overrides", "kind", "created_at",
        )}
```

> 漏掉这一处，旧库第一次触发重建时 `kind` 会被静默丢弃 —— 没有报错，没有日志，只是这个列消失了。Step 1 的第三个测试就是钉这一行的。

- [ ] **Step 5: 让序列化带上 `kind`**

`server/api/avatars.py` 的 `_serialize()` 里加上 `"kind": a.kind`。这一步做完 Step 1 的第一个测试才会转绿。

- [ ] **Step 6: 运行，确认通过**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_kind_migration.py -v
```
预期：3 个测试全 PASS。

- [ ] **Step 7: 确认没有打破现有测试**

```
.venv/Scripts/python.exe -m pytest
```
预期：全绿，且仍在十秒内（torch-free）。

- [ ] **Step 8: 提交**

```bash
git add server/db/models.py server/db/session.py server/api/avatars.py tests/test_live2d_kind_migration.py
git commit -m "feat(db): add Avatar.kind so live2d and photoreal coexist in one table"
```

---

### Task 7: Live2D 模型注册校验

**Files:**
- Create: `server/live2d/__init__.py`, `server/live2d/registry.py`
- Test: `tests/test_live2d_registry.py`

**Interfaces:**
- Consumes: `Avatar.kind`（Task 6）
- Produces:
  ```python
  from server.live2d.registry import validate_model_dir, Live2DModelError
  validate_model_dir(path: Path, config: dict) -> dict
      # 校验通过返回规范化后的 config；失败抛 Live2DModelError
      #   Live2DModelError(code: str, message: str)
  ```

- [ ] **Step 1: 写失败的测试**

```python
import json
from pathlib import Path

import pytest

from server.core.errors import AppError
from server.live2d.registry import validate_model_dir


def _write_model(root: Path, *, with_moc: bool = True, expr_ids=("0", "1")) -> Path:
    model = root / "model.model3.json"
    model.write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {
                    "Moc": "model.moc3" if with_moc else "missing.moc3",
                    "Textures": ["texture_00.png"],
                    "Expressions": [
                        {"Name": e, "File": f"{e}.exp3.json"} for e in expr_ids
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    if with_moc:
        (root / "model.moc3").write_bytes(b"\x00" * 16)
    (root / "texture_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    for e in expr_ids:
        (root / f"{e}.exp3.json").write_text("{}", encoding="utf-8")
    return root


def test_accepts_a_complete_model(tmp_path):
    cfg = validate_model_dir(_write_model(tmp_path), {"model": {"entry": "model.model3.json"}})
    assert cfg["model"]["entry"] == "model.model3.json"


def test_rejects_a_missing_entry_file(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(tmp_path, {"model": {"entry": "nope.json"}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_moc(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(_write_model(tmp_path, with_moc=False), {"model": {"entry": "model.model3.json"}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_missing_texture(tmp_path):
    root = _write_model(tmp_path)
    (root / "texture_00.png").unlink()
    with pytest.raises(AppError) as ei:
        validate_model_dir(root, {"model": {"entry": "model.model3.json"}})
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_an_expression_id_not_in_the_model(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(
            _write_model(tmp_path, expr_ids=("0",)),
            {"model": {"entry": "model.model3.json"}, "expressions": {"smile": 7}},
        )
    assert ei.value.code == "live2d_invalid_model"


def test_rejects_a_path_traversal_entry(tmp_path):
    with pytest.raises(AppError) as ei:
        validate_model_dir(tmp_path, {"model": {"entry": "../../etc/passwd"}})
    assert ei.value.code == "live2d_invalid_model"
```

- [ ] **Step 2: 运行，确认失败**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_registry.py -v
```
预期：collection ERROR，`ModuleNotFoundError: No module named 'server.live2d'`。

- [ ] **Step 3: 实现**

`server/live2d/__init__.py` 留空。`server/live2d/registry.py`：

```python
"""Validate a Live2D Cubism project directory at registration time.

Registering is the only moment we can cheaply check the model is whole.
A Cubism project is a dozen interlinked files (``model3.json`` names the
moc3, the textures, every expression and every motion); a missing one
otherwise surfaces as a blank canvas in the browser, minutes later, with
no server-side trace.
"""

import json
from pathlib import Path

from server.core.errors import AppError

_BAD = "live2d_invalid_model"

_ENTRY_KEYS = ("Moc", "Textures", "Expressions", "Motions")


def _fail(message: str) -> AppError:
    return AppError(_BAD, message, 422)


def _resolve(root: Path, relative: str) -> Path:
    """Resolve a model-relative path, refusing anything that escapes root."""
    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        raise _fail(f"illegal path in model: {relative!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise _fail(f"illegal path in model: {relative!r}")
    return target


def validate_model_dir(path: Path, config: dict) -> dict:
    root = Path(path)
    entry_name = (config.get("model") or {}).get("entry")
    if not entry_name:
        raise _fail("config.model.entry is required")
    entry = _resolve(root, entry_name)
    if not entry.is_file():
        raise _fail(f"entry file not found: {entry_name}")

    try:
        spec = json.loads(entry.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _fail(f"entry file is not readable JSON: {exc}") from exc

    refs = spec.get("FileReferences") or {}
    moc = refs.get("Moc")
    if not moc:
        raise _fail("entry file declares no Moc")
    if not _resolve(root, moc).is_file():
        raise _fail(f"missing moc: {moc}")

    for texture in refs.get("Textures") or []:
        if not _resolve(root, texture).is_file():
            raise _fail(f"missing texture: {texture}")

    model_expr_ids = set()
    for expr in refs.get("Expressions") or []:
        name = str(expr.get("Name"))
        model_expr_ids.add(name)
        target = _resolve(root, expr.get("File", ""))
        if not target.is_file():
            raise _fail(f"missing expression file: {expr.get('File')}")

    for group in (refs.get("Motions") or {}).values():
        for motion in group or []:
            if not _resolve(root, motion.get("File", "")).is_file():
                raise _fail(f"missing motion file: {motion.get('File')}")

    # The operator's emotion map indexes into the model's own expression
    # list. An off-by-one here is invisible until the pet tries to make a
    # face that does not exist.
    for tag, idx in (config.get("expressions") or {}).items():
        if str(idx) not in model_expr_ids:
            raise _fail(
                f"expressions[{tag!r}] = {idx} but the model declares {sorted(model_expr_ids)}"
            )

    return config
```

- [ ] **Step 4: 运行，确认通过**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_registry.py -v
```
预期：6 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
git add server/live2d/ tests/test_live2d_registry.py
git commit -m "feat(live2d): validate Cubism projects at registration time"
```

---

### Task 8: 模型托管与形态过滤

**Files:**
- Modify: `server/api/avatars.py`（列表加 `kind` 过滤；新增模型文件端点）
- Create: `server/api/client.py`（新路由的骨架，本任务先建空壳）
- Modify: `server/main.py`（挂载 `/live2d`）
- Test: `tests/test_live2d_assets.py`

**Interfaces:**
- Consumes: `require_v1_token`（Task 9 建立）、`Avatar.kind`（Task 6）
- Produces:
  - `GET /api/v1/avatars?kind=live2d`
  - `GET /api/v1/avatars/{avatar_id}/model/{path:path}` —— 带 `X-V1-Token`

> 端点顺序：Task 9 会往 `server/api/client.py` 加路由。先建本任务的资产端点，Task 9 再加两个对话端点。

- [ ] **Step 1: 写失败的测试**

```python
import json

from sqlalchemy.orm import sessionmaker

from server.db.models import Avatar
from server.db.session import get_engine

# Must be uuid4 hex: ``_require_uuid_hex`` rejects anything else before the
# request ever reaches the filesystem.
LIVE2D_ID = "a" * 32


def _seed_live2d(client):
    """Register one live2d avatar by writing its row directly -- the upload
    path is admin-only and the point here is the read side."""
    root = client.app.state.session_manager.store.avatar_dir(LIVE2D_ID)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.model3.json").write_text(
        json.dumps({"Version": 3, "FileReferences": {"Moc": "model.moc3", "Textures": []}}),
        encoding="utf-8",
    )
    (root / "model.moc3").write_bytes(b"\x00" * 16)

    db = sessionmaker(bind=get_engine())()
    try:
        db.add(
            Avatar(
                id=LIVE2D_ID,
                name="l2d",
                status="ready",
                source_image_path="",
                kind="live2d",
                config_json={"model": {"entry": "model.model3.json"}},
            )
        )
        db.commit()
    finally:
        db.close()


def _enable_token(monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret")
    from server.config import get_settings

    get_settings.cache_clear()


def test_default_listing_excludes_live2d(client):
    _seed_live2d(client)
    ids = [a["avatar_id"] for a in client.get("/api/v1/avatars").json()]
    assert LIVE2D_ID not in ids


def test_kind_filter_returns_live2d(client):
    _seed_live2d(client)
    resp = client.get("/api/v1/avatars", params={"kind": "live2d"})
    assert [a["avatar_id"] for a in resp.json()] == [LIVE2D_ID]


def test_model_files_require_the_v1_token(client, monkeypatch):
    _seed_live2d(client)
    _enable_token(monkeypatch)
    resp = client.get(f"/api/v1/avatars/{LIVE2D_ID}/model/model.model3.json")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_model_files_are_served_with_the_token(client, monkeypatch):
    _seed_live2d(client)
    _enable_token(monkeypatch)
    resp = client.get(
        f"/api/v1/avatars/{LIVE2D_ID}/model/model.model3.json",
        headers={"X-V1-Token": "s3cret"},
    )
    assert resp.status_code == 200
    assert resp.json()["Version"] == 3


def test_model_path_traversal_is_refused(client, monkeypatch):
    _seed_live2d(client)
    _enable_token(monkeypatch)
    resp = client.get(
        f"/api/v1/avatars/{LIVE2D_ID}/model/../../../secrets.env",
        headers={"X-V1-Token": "s3cret"},
    )
    assert resp.status_code in (400, 404)
```

- [ ] **Step 2: 运行，确认失败**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_assets.py -v
```
预期：`ModuleNotFoundError: server.core.auth`。

- [ ] **Step 3: 先建 `require_v1_token`（Task 9 也要用）**

`server/config.py` 的 `Settings` 加 `v1_api_token: str = ""`。

`server/core/auth.py`：

```python
"""Token gate for the business-plane client endpoints.

The existing ``/api/v1`` routes are unauthenticated after the no-tenant
refactor (see spec §2), and this module deliberately does not change that.
It guards only the endpoints added for the Live2D client, which are the
first ``/api/v1`` endpoints a *remote* caller will reach and the first
that spend upstream LLM/TTS quota. Retrofitting the existing routes is a
separate piece of work with its own blast radius.
"""

import secrets

from fastapi import Header

from server.config import get_settings
from server.core.errors import AppError


async def require_v1_token(x_v1_token: str | None = Header(default=None)) -> None:
    """Pass when no token is configured, or when the header matches.

    Unset (``DH_V1_API_TOKEN=""``) disables the check so local development
    needs no configuration. A configured token means every caller must
    present it, including the admin console if it ever calls these.
    """
    expected = get_settings().v1_api_token
    if not expected:
        return
    if not x_v1_token or not secrets.compare_digest(x_v1_token, expected):
        raise AppError("unauthorized", "missing or invalid X-V1-Token", 401)
```

- [ ] **Step 4: 加模型文件端点与 kind 过滤**

`server/api/avatars.py`：列表端点加 `kind: str | None = None` 查询参数，`kind` 非空时加 `.where(Avatar.kind == kind)`。新增：

```python
@router.get("/{avatar_id}/model/{path:path}")
def get_model_file(
    avatar_id: str,
    path: str,
    db=Depends(get_db),
    _token=Depends(require_v1_token),
):
    """Serve one file from a live2d model's directory.

    Authenticated because a model directory is a configured asset, not
    public content -- an unauthenticated read here would let anyone enumerate
    and download the deployment's model library. The avatar id is uuid4 hex
    and the path is re-resolved under the avatar root, so neither can walk
    out of the tree.
    """
    _require_uuid_hex(avatar_id)
    row = db.get(Avatar, avatar_id)
    if row is None or row.kind != "live2d":
        raise AppError("not_found", f"avatar {avatar_id} not found", 404)
    root = _store().avatar_dir(avatar_id).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise AppError("not_found", "model file not found", 404)
    return FileResponse(target)
```

> `_store().avatar_dir(id)` 是 `AvatarStore` 上已有的方法（`server/avatar/store.py:18`），返回 `data_dir/avatars/<id>`，不需要新增。

- [ ] **Step 5: 挂载 `/live2d`**

`server/main.py` 里照抄 `/admin` 的挂法：

```python
    l2d_dist = Path(__file__).resolve().parent.parent / "live2d-ui" / "dist"
    if l2d_dist.exists():
        app.mount("/live2d", SPAStaticFiles(directory=str(l2d_dist), html=True), name="live2d")
```

- [ ] **Step 6: 运行，确认通过**

```
.venv/Scripts/python.exe -m pytest tests/test_live2d_assets.py -v
```
预期：5 个测试全 PASS。

- [ ] **Step 7: 提交**

```bash
git add server/core/auth.py server/config.py server/api/avatars.py server/main.py tests/test_live2d_assets.py
git commit -m "feat(api): serve live2d model files behind a token and filter avatars by kind"
```

---

### Task 9: 业务面 LLM / TTS 端点

**Files:**
- Create: `server/api/client.py`
- Modify: `server/main.py`（include_router）
- Test: `tests/test_client_endpoints.py`

**Interfaces:**
- Consumes: `require_v1_token`（Task 8）、`stream_chat`、`synthesize`
- Produces:
  - `POST /api/v1/chat/stream` → `text/event-stream`，帧 `{"delta"}` / `{"done": true}` / `{"error": {...}}`
  - `POST /api/v1/tts` → `audio/wav`

- [ ] **Step 1: 写失败的测试**

```python
import json

import httpx


def test_chat_stream_requires_the_token(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret")
    from server.config import get_settings
    get_settings.cache_clear()
    resp = client.post("/api/v1/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_tts_requires_the_token(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "s3cret")
    from server.config import get_settings
    get_settings.cache_clear()
    resp = client.post("/api/v1/tts", json={"text": "hi", "voice_id": "v"})
    assert resp.status_code == 401


def test_chat_stream_emits_delta_then_done(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "")
    from server.config import get_settings
    get_settings.cache_clear()

    async def fake_stream(messages, **kwargs):
        yield "[smile]你好"
        yield "呀"

    monkeypatch.setattr("server.llm.openai.stream_chat", fake_stream)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: "):])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames[0] == {"delta": "[smile]你好"}
    assert frames[1] == {"delta": "呀"}
    assert frames[-1] == {"done": True}


def test_chat_stream_maps_upstream_failure_to_an_error_frame(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "")
    from server.config import get_settings
    get_settings.cache_clear()

    from server.core.errors import AppError

    async def boom(messages, **kwargs):
        raise AppError("llm_not_configured", "no llm", 503)
        yield  # pragma: no cover

    monkeypatch.setattr("server.llm.openai.stream_chat", boom)
    resp = client.post(
        "/api/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200      # the stream is already committed
    assert '"error"' in resp.text


def test_tts_rejects_empty_text(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "")
    from server.config import get_settings
    get_settings.cache_clear()
    resp = client.post("/api/v1/tts", json={"text": "   ", "voice_id": "v"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_text"


def test_tts_returns_wav_bytes(client, monkeypatch):
    monkeypatch.setenv("DH_V1_API_TOKEN", "")
    from server.config import get_settings
    get_settings.cache_clear()

    async def fake_synth(text, voice_id, **kwargs):
        return b"RIFF....WAVEfake"

    monkeypatch.setattr("server.tts.minimax.synthesize", fake_synth)
    resp = client.post("/api/v1/tts", json={"text": "hi", "voice_id": "v"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content.startswith(b"RIFF")
```

- [ ] **Step 2: 运行，确认失败**

```
.venv/Scripts/python.exe -m pytest tests/test_client_endpoints.py -v
```
预期：全部 404（路由还不存在）。

- [ ] **Step 3: 实现**

`server/api/client.py`：

```python
"""Business-plane endpoints for the Live2D client.

Same provider code as the admin console (``server/llm/openai.py``,
``server/tts/minimax.py``) reached through a different surface: these are
called by a product client over the network rather than by a single-operator
debug console on localhost, so they carry the ``X-V1-Token`` gate.
"""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from server.core.auth import require_v1_token
from server.core.errors import AppError
from server.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["client"])


class LlmChatIn(BaseModel):
    model_config = {"extra": "forbid"}
    messages: list[dict[str, str]]


class TtsIn(BaseModel):
    model_config = {"extra": "forbid"}
    text: str
    voice_id: str


@router.post("/chat/stream")
async def chat_stream(
    body: LlmChatIn,
    db=Depends(get_db),
    _token=Depends(require_v1_token),
):
    """Streaming chat. Same frame shape as the admin console's endpoint so
    the client can share one SSE parser with it.

    Errors arrive as an in-band ``{"error": ...}`` frame rather than an HTTP
    status: by the time an upstream failure surfaces, the 200 and its headers
    are already on the wire.
    """
    from server.llm.openai import stream_chat

    async def event_source():
        try:
            async for delta in stream_chat(body.messages, db=db):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except AppError as exc:
            yield f"data: {json.dumps({'error': {'code': exc.code, 'message': exc.message}})}\n\n"
            return
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/tts")
async def synthesize_speech(
    body: TtsIn,
    db=Depends(get_db),
    _token=Depends(require_v1_token),
):
    """Text to WAV. The bytes come back exactly as the provider produced them
    -- resampling stays in the browser, so the server grows no audio codec
    dependency (see spec §4.3)."""
    text = body.text.strip()
    if not text:
        raise AppError("empty_text", "text must not be empty", 400)
    from server.tts.minimax import synthesize

    audio = await synthesize(text, body.voice_id, db=db)
    return Response(audio, media_type="audio/wav")
```

- [ ] **Step 4: 注册路由**

`server/main.py` 的 `create_app()` 里，紧挨现有的 router 导入加：

```python
    from server.api.client import router as client_router
    ...
    app.include_router(client_router)
```

- [ ] **Step 5: 运行，确认通过**

```
.venv/Scripts/python.exe -m pytest tests/test_client_endpoints.py -v
```
预期：6 个测试全 PASS。

- [ ] **Step 6: 全量回归**

```
.venv/Scripts/python.exe -m pytest
```
预期：全绿。

- [ ] **Step 7: 提交**

```bash
git add server/api/client.py server/main.py tests/test_client_endpoints.py
git commit -m "feat(api): add token-gated business-plane LLM and TTS endpoints"
```

---

## Phase 3 — 接线与桌面外壳

### Task 10: 客户端 API 层与句子队列

**Files:**
- Create: `live2d-ui/src/api/sentences.ts`, `live2d-ui/src/api/sentences.test.ts`
- Create: `live2d-ui/src/api/client.ts`, `live2d-ui/src/api/client.test.ts`

**Interfaces:**
- Consumes: `/api/v1/chat/stream`、`/api/v1/tts`
- Produces:
  ```ts
  export const EMOTION_TAG = /\[(\w+)\]/g
  export function stripEmotionTags(text: string): { text: string; tags: string[] }
  export function splitSentences(buffer: string, push: string): { complete: string[]; tail: string }

  export interface ClientConfig { baseUrl: string; token: string }
  export async function* streamChat(cfg: ClientConfig, messages: {role: string; content: string}[]): AsyncGenerator<string>
  export async function synthesize(cfg: ClientConfig, text: string, voiceId: string): Promise<ArrayBuffer>
  ```

- [ ] **Step 1: 写失败的测试**

`sentences.test.ts` —— **Review Focus #1 就在这里**：

```ts
import { describe, expect, it } from "vitest"
import { splitSentences, stripEmotionTags } from "./sentences"

describe("stripEmotionTags", () => {
  it("removes the tag from the text sent to TTS", () => {
    // A tag left in would be read aloud as literal text.
    const r = stripEmotionTags("[smile]你好呀")
    expect(r.text).toBe("你好呀")
    expect(r.tags).toEqual(["smile"])
  })

  it("collects several tags in order", () => {
    expect(stripEmotionTags("[smile]a[think]b").tags).toEqual(["smile", "think"])
  })

  it("leaves ordinary brackets alone", () => {
    expect(stripEmotionTags("see [1] and [note]").text).toBe("see [1] and [note]")
  })

  it("keeps CJK punctuation as a sentence boundary", () => {
    const r = splitSentences("", "[smile]你好。再见")
    expect(r.complete).toEqual(["[smile]你好。"])
    expect(r.tail).toBe("再见")
  })
})

describe("splitSentences", () => {
  it("holds the partial sentence as tail", () => {
    expect(splitSentences("", "你好").complete).toEqual([])
  })

  it("does not emit an empty sentence", () => {
    const r = splitSentences("", "。")
    expect(r.complete).toEqual(["。"])
    expect(r.tail).toBe("")
  })
})
```

`client.test.ts`：

```ts
import { describe, expect, it } from "vitest"
import { parseSseFrames } from "./client"

describe("parseSseFrames", () => {
  it("reads delta frames", () => {
    expect(parseSseFrames('data: {"delta":"a"}\n\ndata: {"delta":"b"}\n\n'))
      .toEqual([{ delta: "a" }, { delta: "b" }])
  })

  it("reads the done frame", () => {
    expect(parseSseFrames('data: {"done":true}\n\n')).toEqual([{ done: true }])
  })

  it("reads an in-band error frame", () => {
    expect(parseSseFrames('data: {"error":{"code":"x","message":"y"}}\n\n'))
      .toEqual([{ error: { code: "x", message: "y" } }])
  })

  it("ignores blank and comment lines", () => {
    expect(parseSseFrames(': keep-alive\n\ndata: {"delta":"a"}\n\n')).toEqual([{ delta: "a" }])
  })
})
```

- [ ] **Step 2: 运行，确认失败**

`npx vitest run src/api/` → FAIL。

- [ ] **Step 3: 实现 `sentences.ts`**

```ts
/** Emotion tags ride inline in the LLM delta stream, e.g. "[smile]好的呀".
 *  They must be stripped BEFORE the text reaches TTS -- otherwise the pet
 *  says the tag out loud. */
const TAG = /\[([a-z]+)\]/gi

const BOUNDARY = /[。！？!?；;\n]/

export function stripEmotionTags(text: string): { text: string; tags: string[] } {
  const tags: string[] = []
  const stripped = text.replace(TAG, (_m, name: string) => {
    tags.push(name.toLowerCase())
    return ""
  })
  return { text: stripped, tags }
}

export function splitSentences(
  buffer: string,
  push: string,
): { complete: string[]; tail: string } {
  const merged = buffer + push
  const complete: string[] = []
  let last = 0
  for (let i = 0; i < merged.length; i++) {
    if (BOUNDARY.test(merged[i])) {
      const piece = merged.slice(last, i + 1).trim()
      if (piece) complete.push(piece)
      last = i + 1
    }
  }
  return { complete, tail: merged.slice(last) }
}
```

- [ ] **Step 4: 实现 `client.ts`**

```ts
export interface ClientConfig {
  baseUrl: string
  token: string
}

export interface SseFrame {
  delta?: string
  done?: boolean
  error?: { code: string; message: string }
}

export function parseSseFrames(raw: string): SseFrame[] {
  const out: SseFrame[] = []
  for (const line of raw.split("\n")) {
    if (!line.startsWith("data: ")) continue
    try {
      out.push(JSON.parse(line.slice(6)) as SseFrame)
    } catch {
      // A truncated frame from a dropped connection. Skip it; the caller
      // reconnects rather than failing the whole turn.
    }
  }
  return out
}

function headers(cfg: ClientConfig): HeadersInit {
  return { "Content-Type": "application/json", "X-V1-Token": cfg.token }
}

export async function* streamChat(
  cfg: ClientConfig,
  messages: { role: string; content: string }[],
): AsyncGenerator<string> {
  const resp = await fetch(`${cfg.baseUrl}/api/v1/chat/stream`, {
    method: "POST",
    headers: headers(cfg),
    body: JSON.stringify({ messages }),
  })
  if (resp.status === 401) throw new Error("unauthorized")
  if (!resp.ok || !resp.body) throw new Error(`chat stream failed: ${resp.status}`)
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const frame of parseSseFrames(chunk)) {
        if (frame.error) throw new Error(frame.error.message)
        if (frame.delta) yield frame.delta
      }
    }
  }
}

export async function synthesize(
  cfg: ClientConfig,
  text: string,
  voiceId: string,
): Promise<ArrayBuffer> {
  const resp = await fetch(`${cfg.baseUrl}/api/v1/tts`, {
    method: "POST",
    headers: headers(cfg),
    body: JSON.stringify({ text, voice_id: voiceId }),
  })
  if (resp.status === 401) throw new Error("unauthorized")
  if (!resp.ok) throw new Error(`tts failed: ${resp.status}`)
  return resp.arrayBuffer()
}
```

- [ ] **Step 5: 运行，确认通过**

`npx vitest run` → 全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add live2d-ui/src/api/
git commit -m "feat(live2d-ui): add the token client, SSE parser and sentence queue"
```

---

### Task 11: Electron 桌面外壳

**Files:**
- Create: `live2d-ui/electron/main.cjs`, `live2d-ui/electron/preload.cjs`, `live2d-ui/electron/config.json`
- Modify: `live2d-ui/package.json`（加 `electron`、`electron-builder`、`main` 字段）

**Interfaces:**
- Consumes: `live2d-ui/dist`（Task 5 的构建产物）
- Produces: 桌面宠物；preload 暴露 `window.petConfig.{get,set}`

- [ ] **Step 1: 写主进程**

`live2d-ui/electron/main.cjs`：

```js
const { app, BrowserWindow, Tray, Menu, ipcMain } = require("electron")
const path = require("node:path")
const fs = require("node:fs")

const CONFIG_PATH = path.join(app.getPath("userData"), "pet.json")
let win = null
let tray = null

function readConfig() {
  try {
    return JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"))
  } catch {
    return { serverUrl: "http://127.0.0.1:8000", token: "" }
  }
}

function createWindow() {
  const cfg = readConfig()
  win = new BrowserWindow({
    width: 400,
    height: 600,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    hasShadow: false,
    resizable: false,
    skipTaskbar: true,
    backgroundColor: "#00000000",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // A desktop pet is ALWAYS unfocused -- it is a thing on the background.
      // Without this, Electron pauses requestAnimationFrame for the window
      // and the pet stands completely still. This is the single most
      // important line in this file.
      backgroundThrottling: false,
    },
  })
  win.loadFile(path.join(__dirname, "..", "dist", "index.html"))
  win.setIgnoreMouseEvents(true, { forward: true })
  win.on("mousemove", () => {
    if (win) win.setIgnoreMouseEvents(false)
  })
  win.on("mouseleave", () => {
    if (win) win.setIgnoreMouseEvents(true, { forward: true })
  })
  win.webContents.send("pet:config", cfg)
}

function buildTray() {
  tray = new Tray(path.join(__dirname, "..", "build", "tray.png"))
  tray.setToolTip("Live2D 宠物")
  const refresh = () => tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "服务地址", click: () => editServerUrl() },
      { type: "separator" },
      { label: "显示", click: () => win && win.show() },
      { label: "隐藏", click: () => win && win.hide() },
      { type: "separator" },
      { label: "退出", click: () => app.quit() },
    ]),
  )
  refresh()
}

function editServerUrl() {
  const cfg = readConfig()
  const next = require("electron").dialog.showMessageBoxSync
  void next  // placeholder replaced below
}

app.whenReady().then(() => {
  createWindow()
  buildTray()
  ipcMain.handle("pet:config:set", (_e, patch) => {
    const next = { ...readConfig(), ...patch }
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(next, null, 2))
    return next
  })
})

app.on("window-all-closed", () => app.quit())
```

> `editServerUrl` 上面那段是占位写法，会跑不通。用下面这个真实现替换它 —— 用 `dialog.showInputBox` 的替代方案：Electron 没有内置输入框对话框，所以用一个 `prompt()` 不可用。最简做法是**配置文件 + 重新启动**：托盘菜单项改为「打开配置文件」，用 `shell.openPath(CONFIG_PATH)` 交给系统默认编辑器。**这是本任务唯一需要人工确认的交互选择**，实现时按此定稿，不要留空函数。

- [ ] **Step 2: 写 preload**

`live2d-ui/electron/preload.cjs`：

```js
const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("petConfig", {
  initial: {},
  set: (patch) => ipcRenderer.invoke("pet:config:set", patch),
  onConfig: (cb) => ipcRenderer.on("pet:config", (_e, cfg) => cb(cfg)),
})
```

- [ ] **Step 3: 打包配置**

`live2d-ui/package.json` 加 `"main": "electron/main.cjs"`、devDependency `electron` 与 `electron-builder`，并加 `build` 段产出 Windows 安装包。

- [ ] **Step 4: 人工验证（Playwright 或手工）**

- [ ] 窗口无边框、透明、始终置顶，不出现在任务栏
- [ ] **点击桌面其它位置时宠物仍在动**（Review Focus #4 —— 这条挂了整个产品就没意义了）
- [ ] 鼠标移入宠物区域时可交互，移开后恢复穿透
- [ ] 托盘菜单能打开配置文件，重启后新地址生效
- [ ] 服务地址不可达时，**idle 动画仍在跑**，且界面提示「未连接」
- [ ] `X-V1-Token` 配错时提示鉴权失败，且**不无限重试**

- [ ] **Step 5: 提交**

```bash
git add live2d-ui/electron/ live2d-ui/package.json
git commit -m "feat(live2d-ui): add the Electron desktop-pet shell"
```

---

### Task 12: 端到端接线与文档

**Files:**
- Modify: `live2d-ui/src/App.tsx`（接上真实链路）
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部前序任务
- Produces: 可用的完整产品

- [ ] **Step 1: 接上会话循环**

`App.tsx` 里：输入框 → `streamChat` → `splitSentences` → 队列 → 逐句 `stripEmotionTags`（同时喂 `expressionIndex` 与送 TTS 的干净文本）→ `synthesize` → `decodeAudioData` → `analyzeSpeech` → 播放并按 `currentTime` 查表喂 `applyFrame`。

> 语音预取：起播第 N 句的同时请求第 N+1 句，否则每句都会在播放前留一段等网络的空白。这正是 `admin-ui/src/pages/Chat.tsx` 里已有的做法，照抄其节奏。

- [ ] **Step 2: 实现失败降级**

SSE 断开 → 退避重连，idle 不停；TTS 失败 → 跳过该句、继续队列、字幕照常；401 → 停止重试并提示配置；服务端不可达 → 提示但 idle 继续。

- [ ] **Step 3: 端到端人工验证**

用真实模型跑一遍：打字 → 宠物说话 → 嘴动且同步 → 表情切换 → 说话结束后回到 idle 且**呼吸相位连续无跳变**。

- [ ] **Step 4: 修 README**

README 里的 `X-API-Key` 描述是过时的（鉴权已在 no-tenant 重构中删除）。改成本次的真实情况，并补上新端点表：

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/api/v1/chat/stream` | `X-V1-Token` | SSE，`{"delta"}` / `{"done"}` / `{"error"}` |
| POST | `/api/v1/tts` | `X-V1-Token` | `{"text","voice_id"}` → `audio/wav` |
| GET | `/api/v1/avatars/{id}/model/{path}` | `X-V1-Token` | Live2D 模型文件 |
| GET | `/api/v1/avatars?kind=live2d` | 无 | 只列 Live2D 形象 |

并补 `DH_V1_API_TOKEN` 到环境变量表。

- [ ] **Step 5: 全量回归**

```
.venv/Scripts/python.exe -m pytest
cd live2d-ui && npx vitest run && npx tsc --noEmit
```
预期：全绿。

- [ ] **Step 6: 提交**

```bash
git add live2d-ui/src/App.tsx README.md
git commit -m "feat(live2d): wire the conversation loop and correct the stale auth docs"
```

---

## Self-Review

**1. Spec 覆盖**

| Spec 节 | 覆盖任务 |
|---|---|
| §3.1 同服务不新开 | Task 6–9（全部在现有服务内） |
| §3.2 口型在客户端 / 不开 session | Task 2, 4 |
| §3.3 一个 bundle 两种外壳 | Task 5（构建）, Task 8（挂载）, Task 11 |
| §3.4 不另开表 | Task 6 |
| §4.1 `kind` 迁移 | Task 6 |
| §4.2 注册校验 + 带鉴权托管 | Task 7, 8 |
| §4.3 业务面 LLM/TTS + token | Task 8（auth）, 9 |
| §4.4 情绪标签 | Task 10（剥离）, 12（prompt 接入） |
| §5.1 离线包络 + 真实参数范围 | Task 2, 4 |
| §5.2 生命感层 | Task 3 |
| §5.3 表情导演 | Task 4 |
| §5.4 帧序 | Task 4 |
| §5.5 生命周期 | Task 3（相位连续）, 12 |
| §6 Electron 四件事 | Task 11 |
| §7 错误降级 | Task 12（+ 各任务的内建降级） |
| §8 测试策略 | 各任务；WebGL 部分为 Task 5/11 的人工验证 |
| §9 SDK 风险 | Task 1 |
| §10 交付顺序 | Phase 0→3 |

无缺口。唯一半自动的是 §4.4 的 prompt 改动（Task 12 Step 1 顺带做，未单列测试）。

**2. 占位符扫描** —— 无 TBD/TODO。唯一的「占位」是 Task 11 Step 1 的 `editServerUrl`，已在同一步内给出定稿方案（改用 `shell.openPath`）并注明不得留空函数。

**3. 类型一致性** —— `IdlePose` 在 Task 3 定义、Task 4 引用、Task 5 消费，字段一致。`analyzeSpeech` 返回 `{open, form}`，Task 12 按 `open`/`form` 取用。`applyFrame` 的 `FrameInputs` 与 Task 4 的测试对象字面量一致。`require_v1_token` 在 Task 8 定义、Task 8/9 共用。

**4. Review Focus 覆盖** —— #1 Task 10、#2 Task 4、#3 Task 4、#4 Task 11 人工验证、#5 Task 2。四条有自动化测试，#4 是 Electron 行为、只能人工验证，已在 Task 11 Step 4 列为显式检查项。