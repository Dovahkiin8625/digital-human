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
export interface ModelInfo {
  id: string;
  // "drive_runtime" is the only flavour the engine exposes today;
  // the field exists so the column can badge the row cleanly when a
  // second kind (portrait_animator, tts_voice, ...) lands.
  type: string;
  // The model parts the runtime materialises. Renders as a Tag each,
  // or as "+N" once there are too many to fit a row. Strings are
  // display hints; not paths.
  components: string[];
  // Absolute path on disk the runtime reads weights from. Null when
  // the engine has no weights (mock).
  source: string | null;
  // Stable, reproducible lower bound: sum of all weight files in GB.
  // The actual VRAM peak is rendered on the GPU snapshot card.
  estimated_bytes: number;
  loaded: boolean;
  // False for engines that hold no loadable weights (mock). Drives the
  // unload button visibility.
  can_unload: boolean;
  // ISO timestamp the runtime last became live (mounted weight).
  // Null when the model is not currently loaded. Used to render
  // "已加载 3 分钟前".
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

// One row per every OTHER model this server uses -- portrait animators
// (LivePortrait / Wan 2.2 / MuseTalk silent) + MiniMax TTS. Lifecycle
// state is surfaced so the operator can drive load / unload from here
// without bouncing to 形象活化 tab or paying mid-call cold-loads.
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
  // (e.g. musetalk-silent piggybacks on the drive engine).
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