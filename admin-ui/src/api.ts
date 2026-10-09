const KEY = "dh_admin_token";

export function getToken(): string | null {
  return localStorage.getItem(KEY);
}

export function setToken(t: string | null): void {
  if (t) localStorage.setItem(KEY, t);
  else localStorage.removeItem(KEY);
}

export async function api<T>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["X-Admin-Token"] = token;
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
    try {
      const data = await resp.json();
      message = data?.error?.message ?? message;
    } catch {
      /* non-json error */
    }
    throw new Error(message);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export function login(password: string): Promise<{ token: string }> {
  return api("/api/admin/login", { method: "POST", body: { password } });
}
