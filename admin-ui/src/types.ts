export interface AvatarInfo {
  avatar_id: string;
  tenant_id: string;
  name: string;
  status: "pending" | "processing" | "ready" | "failed";
  // The pipeline's pack.json: the MuseTalk block plus, for new packs, a
  // top-level `warnings` list of non-blocking source-quality findings.
  config: {
    warnings?: string[];
    [key: string]: unknown;
  };
  error: string | null;
  created_at: string;
}

export interface TenantInfo {
  tenant_id: string;
  name: string;
  created_at: string;
}

export interface SessionInfo {
  session_id: string;
  avatar_id: string;
  status: string;
  metrics: { frames_out?: number; dropped_frames?: number; fps?: number; avg_latency_ms?: number };
}
