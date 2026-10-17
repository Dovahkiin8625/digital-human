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