import type { AvatarInfo, Live2DHealth } from "./types";
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

export async function apiRaw(path: string, opts: { method?: string; body?: unknown } = {}): Promise<Response> {
  const headers: Record<string, string> = {};
  let body: BodyInit | undefined;
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  return fetch(path, { method: opts.method ?? "GET", headers, body });
}

export async function downloadWithToken(path: string): Promise<void> {
  // No auth/token needed after the no-tenant refactor: the browser just fetches
  // and saves the blob. The filename comes from Content-Disposition when
  // present; falls back to the last path segment so the user gets a sensible
  // save name either way.
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

// ----- Live2D -------------------------------------------------------------

export interface Live2DUploadIn {
  name: string;
  /** The whole Cubism project, zipped. */
  file: File;
  /** model3.json's path inside the archive. Blank lets the server detect
   *  it, which is what makes a zip of the exported folder work. */
  entry?: string;
  /** Emotion tag -> index into the model's own expression list. */
  expressions?: Record<string, number>;
}

/**
 * Register a Live2D avatar from an uploaded project archive.
 *
 * Replaces the old server-path registration: an operator used to have to
 * SCP the project onto the box and type an absolute path, which is not a
 * thing a browser can do.
 */
export function uploadLive2dAvatar(body: Live2DUploadIn): Promise<AvatarInfo> {
  const form = new FormData();
  form.append("name", body.name);
  form.append("file", body.file);
  if (body.entry) form.append("entry", body.entry);
  if (body.expressions && Object.keys(body.expressions).length > 0) {
    form.append("expressions", JSON.stringify(body.expressions));
  }
  return api<AvatarInfo>("/api/admin/live2d/avatars", { method: "POST", form });
}

export function deleteLive2dAvatar(avatarId: string): Promise<void> {
  return api<void>(`/api/admin/live2d/avatars/${avatarId}`, { method: "DELETE" });
}

export function live2dHealth(): Promise<Live2DHealth> {
  return api<Live2DHealth>("/api/admin/live2d/health");
}

/**
 * URL of a model's model3.json, which is what <Live2DStage> loads: it
 * resolves the moc3, textures and motions relative to this, so nothing
 * else needs to be configured.
 */
export function live2dModelUrl(avatarId: string, entry: string): string {
  return modelFileUrl(avatarId, entry);
}

/** Base URL of the SDK's Core and Shaders. Not token-gated -- see core.ts. */
export const LIVE2D_RUNTIME_BASE = "/api/v1/live2d/runtime";
/**
 * The business-plane token, as held by the CONSOLE.
 *
 * The admin plane has had no auth since the multi-tenant refactor, so the
 * console has no credential of its own to present. The one exception is the
 * Live2D model host, which is token-gated for the Electron pet's sake -- and
 * which therefore 401s the console too on a deployment that sets one. This
 * is how an operator hands the console the same token the pet uses.
 */
const CONSOLE_TOKEN_KEY = "dhV1Token";

export function getConsoleToken(): string {
  try {
    return window.localStorage.getItem(CONSOLE_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setConsoleToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(CONSOLE_TOKEN_KEY, token);
    else window.localStorage.removeItem(CONSOLE_TOKEN_KEY);
  } catch {
    /* private mode / storage disabled -- the console still works without it */
  }
}
