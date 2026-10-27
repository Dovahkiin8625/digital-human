import type { AvatarInfo, Live2DHealth } from "./types";
import { modelFileUrl } from "./live2d/cubism/core";

/**
 * The pet's client for the digital-human service.
 *
 * Every call is an HTTP request to a server the operator configured: the
 * pet hosts no model files and computes nothing server-side. The token
 * rides along on the business plane's requests -- without it a server that
 * has one configured would 401 on every model fetch and the pet would sit
 * there blank with no explanation.
 */

/** The Live2D avatars the server hosts, ready to be driven. */
export async function listAvatars(baseUrl: string, token: string): Promise<AvatarInfo[]> {
  const root = baseUrl.replace(/\/$/, "");
  const headers: Record<string, string> = {};
  if (token) headers["X-V1-Token"] = token;
  const resp = await fetch(`${root}/api/v1/avatars?drive_model=live2d`, { headers });
  if (!resp.ok) throw new Error(`列出 Live2D 形象失败：${resp.status}`);
  return (await resp.json()) as AvatarInfo[];
}

/**
 * Whether the Cubism SDK is staged on the server.
 *
 * This lives on the admin plane, which a token cannot reach, so a failure
 * here is not fatal -- the runtime route is open. The SDK's Core and
 * Shaders are served from the business plane; only this summary is not.
 */
export async function live2dHealth(baseUrl: string): Promise<Live2DHealth> {
  const root = baseUrl.replace(/\/$/, "");
  const resp = await fetch(`${root}/api/admin/live2d/health`);
  if (!resp.ok) throw new Error(`读取服务端状态失败：${resp.status}`);
  return (await resp.json()) as Live2DHealth;
}

/**
 * URL of a model's model3.json, which is what <Live2DStage> loads: it
 * resolves the moc3, textures and motions relative to this, so nothing
 * else needs to be configured.
 */
export function live2dModelUrl(avatarId: string, entry: string): string {
  return modelFileUrl(avatarId, entry);
}
