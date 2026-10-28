import type {
  AvatarInfo,
  DriveModelInfo,
  Live2DHealth,
  Live2DModelInventory,
} from "./types";
import { modelFileUrl } from "./live2d/cubism/core";

export async function api<T>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  let body: BodyInit | undefined;
  if (opts.form) {
    body = opts.form;
  } else if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, { method: opts.method ?? "GET", headers, body });
  if (!resp.ok) {
    let message = resp.statusText;
    let detail: Record<string, string> | undefined;
    try {
      const data = await resp.json();
      message = data?.error?.message ?? message;
      detail = data?.error?.detail;
    } catch {
      /* non-json error */
    }
    const err = new Error(message) as Error & { detail?: Record<string, string>; status: number };
    err.status = resp.status;
    if (detail) err.detail = detail;
    throw err;
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export async function downloadFile(path: string): Promise<void> {
  // The browser just fetches and saves the blob.
  // The filename comes from Content-Disposition when present; falls back to
  // the last path segment so the user gets a sensible save name either way.
  const resp = await fetch(path);
  if (!resp.ok) {
    let msg = resp.statusText;
    try {
      const data = await resp.json();
      msg = data?.error?.message ?? msg;
    } catch {
      /* non-json error */
    }
    throw new Error(msg);
  }
  const blob = await resp.blob();
  const disp = resp.headers.get("Content-Disposition") ?? "";
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disp);
  const fallback = path.split("/").pop() || "download";
  const filename = match ? decodeURIComponent(match[1]) : fallback;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a tick to start the download before reclaiming the URL.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ----- avatars --------------------------------------------------------------

/**
 * The digital-human models this server can drive an avatar with.
 *
 * One list, shared by the create form and the session pickers: an avatar
 * records the model that owns it, so "which models exist" has a single
 * answer.
 */
export function listDriveModels(): Promise<DriveModelInfo[]> {
  return api<DriveModelInfo[]>("/api/admin/drive-models");
}

interface CreateAvatarIn {
  name: string;
  /** The digital-human model that will drive this avatar. */
  model: string;
  /** The source material: a photo, a clip, or a zipped project. */
  file: File;
  /** Live2D: model3.json's path inside the archive. Blank lets the server
   *  detect it, which is what makes a zip of the exported folder work. */
  entry?: string;
  /** Live2D: emotion tag -> index into the model's own expression list. */
  expressions?: Record<string, number>;
  /** Idle-video knobs, for the models that produce an idle clip. */
  idle_crf?: number;
  idle_preset?: string;
  /** Build options for the model's pipeline, as a JSON object. The models
   *  that need them read their own keys from here; the other models ignore
   *  them. */
  options?: Record<string, unknown>;
}


/**
 * Create an avatar for a digital-human model.
 *
 * One endpoint for every model: the model decides what the upload is (a
 * photo / clip, or a zipped project) and which extra fields apply.
 */
export function createAvatar(body: CreateAvatarIn): Promise<{
  avatar_id: string;
  status: string;
  drive_model: string;
}> {
  const form = new FormData();
  form.append("name", body.name);
  form.append("model", body.model);
  form.append("file", body.file);
  if (body.entry) form.append("entry", body.entry);
  if (body.expressions && Object.keys(body.expressions).length > 0) {
    form.append("expressions", JSON.stringify(body.expressions));
  }
  if (typeof body.idle_crf === "number") form.append("idle_crf", String(body.idle_crf));
  if (body.idle_preset) form.append("idle_preset", body.idle_preset);
  if (body.options && Object.keys(body.options).length > 0) {
    form.append("options", JSON.stringify(body.options));
  }
  return api("/api/admin/avatars", { method: "POST", form });
}

/** Delete an avatar and everything it owns, whichever model drives it. */
export function deleteAvatar(avatarId: string): Promise<void> {
  return api<void>(`/api/admin/avatars/${avatarId}`, { method: "DELETE" });
}

/**
 * Whether the Cubism SDK is staged.
 *
 * The console is served from the same origin as the API, so this is a
 * relative path -- there is no server root to configure.
 */
export function live2dHealth(): Promise<Live2DHealth> {
  return api<Live2DHealth>("/api/admin/live2d/health");
}

/**
 * What the model itself declares: its expression names and motion groups.
 *
 * The emotion map is written in terms of positions in the model's
 * expression list, so the console needs the names behind those positions
 * to offer a choice. Read from the project on disk rather than from the
 * stored config, so an avatar uploaded without a map can still be given
 * one.
 */
export function live2dModelInventory(avatarId: string): Promise<Live2DModelInventory> {
  return api<Live2DModelInventory>(`/api/admin/avatars/${avatarId}/live2d/model`);
}

/** Change a Live2D avatar's emotion map. */
export function updateLive2DConfig(
  avatarId: string,
  body: { expressions?: Record<string, number>; entry?: string },
): Promise<AvatarInfo> {
  return api<AvatarInfo>(`/api/admin/avatars/${avatarId}/live2d`, {
    method: "PATCH",
    body,
  });
}

/**
 * URL of a model's model3.json, which is what <Live2DStage> loads: it
 * resolves the moc3, textures and motions relative to this, so nothing
 * else needs to be configured.
 */
export function live2dModelUrl(avatarId: string, entry: string): string {
  return modelFileUrl(avatarId, entry);
}
