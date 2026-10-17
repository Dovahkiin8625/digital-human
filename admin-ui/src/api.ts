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