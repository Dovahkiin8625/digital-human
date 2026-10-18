export interface AvatarInfo {
  avatar_id: string;
  name: string;
  status: "pending" | "processing" | "ready" | "failed";
  // The pipeline's pack.json: the MuseTalk block plus, for new packs, a
  // top-level `warnings` list of non-blocking source-quality findings.
  config: {
    warnings?: string[];
    // _effective_params is the audit row of TUNABLES actually applied to this
    // avatar at build time. May be absent on legacy / mock packs.
    _effective_params?: TunableParams;
    [key: string]: unknown;
  };
  error: string | null;
  created_at: string;
}

export interface SessionMetrics {
  frames_out?: number;
  dropped_frames?: number;
  fps?: number;
  avg_latency_ms?: number;
  keepalive_frames?: number;
  blocks?: number;
  queue_depth?: number;
  overflows?: number;
  p50_latency_ms?: number;
  p95_latency_ms?: number;
  max_latency_ms?: number;
  // Bucket labels mirror server-side _histogram (e.g. "<25", "25-50", ...).
  latency_histogram?: Record<string, number>;
}

export interface SessionInfo {
  session_id: string;
  avatar_id?: string;
  status: string;
  started_at?: string | null;
  closed_at?: string | null;
  metrics: SessionMetrics;
}

export interface JobInfo {
  job_id: string;
  avatar_id: string;
  stage: string;
  progress: number;
  error: string | null;
  created_at: string;
}

// Engine description returned inside SystemInfo.engine. The mock reports a
// subset; the real engine reports all of them.
export interface EngineInfo {
  name: string;
  gpu: boolean;
  // Only the real engine reports these; the mock has no weights to load.
  models_dir?: string;
  weights_present?: boolean;
  loaded?: boolean;
  avatars_cached?: number;
  pending?: number;
}

// System-wide info from /api/admin/system.
export interface SystemInfo {
  engine: EngineInfo;
  pipeline: string;
  version: string;
  data_dir: string;
  gpu: boolean;
  // Media + session configuration knobs mirrored from server-side config.
  frame_width?: number;
  frame_height?: number;
  fps?: number;
  sample_rate?: number;
  video_bitrate?: number;
  video_max_bitrate?: number;
  session_idle_timeout?: number;
  session_sweep_interval?: number;
  avatar_max_video_seconds?: number;
  avatar_max_upload_bytes?: number;
  minimax_voice_id?: string;
  minimax_sample_rate?: number;
}

// Idle-video knobs the create form exposes per avatar. Bounds mirror the
// idle_crf / idle_preset entries in server/avatar/tunables.py; the
// pipeline still records the full effective dict (with all TUNABLES) in
// _effective_params for audit, but only these two are editable from the UI.
export interface TunableParams {
  idle_crf: number;
  idle_preset: string;
}

// One row from /api/admin/tasks. The backend serialises a GenerationTask
// into this shape; the UI treats it as the source of truth for what's
// currently running and what's just finished.
export interface GenerationTask {
  task_id: string;
  type: string;
  // queued | running | done | failed
  status: "queued" | "running" | "done" | "failed" | string;
  // Human-readable progress hint ("encoding", "done", "loaded weights", ...)
  stage: string;
  progress: number;
  // The free-form payload the operator submitted. For type="i2v" this is
  // { backend, prompt, num_frames, steps, driver, filename }.
  params: Record<string, unknown>;
  // Result payload. For type="i2v" this is { mp4_path } (server-internal);
  // the actual bytes travel back through /api/admin/tasks/{id}/result.mp4.
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

// Pipeline metadata for one avatar (mode + detector + frame counts, ...).
export interface PipelineTimeline {
  mode?: "image" | "video" | string;
  fps?: number;
  source_fps?: number;
  frame_count?: number;
  crop?: unknown;
  detector?: string;
  version?: string;
  warnings?: string[];
  kept_count?: number;
}

// One artifact row from /avatars/{id}/artifacts.
export interface ArtifactInfo {
  path: string;
  size: number;
  kind: "image" | "json" | "tensor" | "mask" | "video" | "binary" | string;
}

// One slot from /api/admin/models. Today there is always exactly one
// (the singleton drive-engine runtime), but the shape is a list so
// adding more slots in future is non-breaking.
//
// One row per thing the operator can put on the GPU. The Models tab
// renders engine slots (``drive_runtime``) and portrait animators
// (``portrait_animator``) in the same table; the optional fields are
// populated by whichever kind produces the row, and the UI falls back
// to ``—`` when a field is null. TTS lives in a separate ``services``
// table because it has no on-GPU weights.
export interface ModelInfo {
  id: string;
  // Drives the 类型 badge + the row grouping. Engine rows report
  // "drive_runtime"; animator rows report "portrait_animator".
  kind: "drive_runtime" | "portrait_animator";
  // Operator-facing label (animator ``meta().label_zh``, or the
  // engine's hardcoded Chinese name). The id column falls back to ``id``
  // when this is absent.
  label_zh?: string;
  // Historical "type" string -- drive_runtime rows still carry this
  // for the 类型 column's tag. Animator rows use ``kind``'s value here
  // so the existing renderer doesn't see a null.
  type: string;
  // The model parts the runtime materialises. Renders as a Tag each,
  // or as "+N" once there are too many to fit a row. Strings are
  // display hints; not paths. Animator rows have an empty array.
  components: string[];
  // Absolute path on disk the runtime reads weights from. Null when
  // the engine has no weights (mock). Animator rows fill this with
  // their own weights_dir.
  source: string | null;
  // Stable, reproducible lower bound: sum of all weight files in GB.
  // The actual VRAM peak is rendered on the GPU snapshot card.
  estimated_bytes: number;
  // Animator-specific extras (all optional; the engine row leaves them
  // undefined and the UI renders blanks).
  group_zh?: string;
  description?: string;
  weights_dir?: string | null;
  weights_present?: boolean;
  missing_script?: string | null;
  vram_required_gb?: number | null;
  vram_hint?: string;
  fits_current_gpu?: boolean | null;
  loaded: boolean;
  // The load button is gated on ``can_load``, NOT on ``can_unload``.
  // ``can_load`` is True when the row exposes a ``load()`` method AND
  // the row isn't currently loaded; ``can_unload`` is True when the
  // row exposes an ``unload()`` method AND the row is currently loaded.
  // The split matters: after an ``unload()`` cycle, ``can_unload`` is
  // False (nothing to release) but ``can_load`` must come back on so
  // the operator can reload. ``can_load`` does NOT require weights on
  // disk -- the click surfaces a 503 with the fetch-script hint.
  can_load: boolean;
  // False for engines that hold no unloadable weights (mock). Drives
  // the unload button visibility.
  can_unload: boolean;
  // ISO timestamp the runtime last became live (mounted weight).
  // Null when the model is not currently loaded. Used to render
  // "已加载 3 分钟前". Animator rows are always null (animators don't
  // track this).
  loaded_at: string | null;
}

// Top-level shape the Models tab polls. gpu is null on a CPU-only
// host (the page renders a "CPU" placeholder instead of a bar).
export interface ModelsPayload {
  models: ModelInfo[];
  services: ServiceInfo[];
  gpu: GpuSnapshot | null;
  active_sessions: number;
}

// MiniMax TTS configuration surfaced by ``GET /api/admin/tts/config`` and
// edited via ``PUT /api/admin/tts/config`` from the Models tab card.
//
// The ``api_key`` is *never* returned in the GET body (only the masked
// form leaks to the UI). ``PUT`` accepts an optional ``api_key``; an
// empty value means "clear the credential" (the synthesise endpoint
// then returns 503 until a real key is saved).
export interface TtsConfigInfo {
  voice_id: string;
  model: string;
  sample_rate: number;
  group_id: string;
  base_url: string;
  api_key_masked: string;
}

export interface TtsConfigUpdate {
  api_key?: string;
  group_id?: string;
  base_url?: string;
  model?: string;
  voice_id?: string;
  sample_rate?: number;
}

// One row per every OTHER model this server uses -- portrait animators
// (LivePortrait / Wan 2.2) + MiniMax TTS. Lifecycle state is surfaced
// so the operator can drive load / unload from here without bouncing
// to 形象活化 tab or paying mid-call cold-loads.
export interface ServiceInfo {
  id: string;
  // ``portrait_animator`` or ``tts_service``. Drives which columns +
  // actions render; same id-shape lets the UI key by id.
  kind: "portrait_animator" | "tts_service" | string;
  label_zh: string;
  group_zh?: string;
  description?: string;
  // True when the service is ready to be invoked (animators: weights
  // + runtime deps; TTS: API key configured).
  configured: boolean;
  // True when the on-disk weights directory exists with content.
  // Always false for TTS (remote service, no on-disk weights).
  weights_present: boolean;
  // Path to the weights directory on disk; null when N/A.
  weights_dir: string | null;
  // For not-ready backends, the fetch script the operator should run
  // to fix it. Null for ready rows.
  missing_script: string | null;
  // VRAM budget the service needs; null means "shares another budget"
  // (e.g. backends that piggyback on a separately-loaded runtime).
  vram_required_gb: number | null;
  vram_hint: string;
  // True / false / null -- null means unknown (no CUDA on this host).
  fits_current_gpu: boolean | null;
  // Whether the pipeline / engine is currently materialised in VRAM.
  // Animators: a per-instance flag. TTS: always false (remote).
  loaded: boolean;
  // True iff the operator can drive a load from the Models tab.
  // False for backends with no in-process weights to mount (mock,
  // TTS-shaped endpoints).
  can_load: boolean;
  // True iff a "卸载" action would do something. Mirrors ``loaded``
  // for the in-process backends.
  can_unload: boolean;
  // Pre-baked label for the status cell: "可加载" / "已加载" / "无权重".
  status: string;
  // TTS-specific extras; absent for animators.
  voice_id?: string;
  model?: string;
  sample_rate?: number;
  sample_rate_cap_hz?: number;
}

// Live per-device counters returned by ``get_vram_snapshot``. All
// numbers in GB, rounded to 3 decimals.
export interface GpuSnapshot {
  device: string;
  name: string;
  total_gb: number;
  allocated_gb: number;
  reserved_gb: number;
  peak_gb: number;
  free_gb: number;
}