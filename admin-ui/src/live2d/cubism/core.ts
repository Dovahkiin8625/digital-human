/** Cubism runtime bootstrap.
 *
 *  The Core and the Shaders come from the server, not from this bundle. They
 *  are Live2D-proprietary, so they are never committed and never shipped in
 *  the build output -- the service serves them from ``DH_LIVE2D_SDK_DIR``,
 *  restricted to the Core/ and Shaders/ subtrees.
 *
 *  That endpoint is deliberately not token-gated: the framework fetches its
 *  own GLSL with a fetch it does not let the caller configure with headers.
 *  Model files, which are operator-supplied, ARE gated, and this module is
 *  also the place a caller declares the token for those.
 */
export interface RuntimeEndpoint {
  /** Absolute server root. Empty in the console (same-origin); required in
   *  Electron, where the document is a file:// page. */
  baseUrl?: string;
  /** Sent as X-V1-Token on every runtime and model request. */
  token?: string;
}

let endpoint: RuntimeEndpoint = {};

/** Point the runtime (and the model fetches in model.ts) at a server.
 *  Called once before the first Live2DStage mounts. */
export function configureRuntime(next: RuntimeEndpoint): void {
  endpoint = { ...endpoint, ...next };
}

function root(): string {
  return endpoint.baseUrl ? endpoint.baseUrl.replace(/\/$/, "") : "";
}

/** Absolute URL for a runtime asset. */
export function runtimeUrl(path: string): string {
  return `${root()}/api/v1/live2d/runtime/${path}`;
}

/** Absolute URL for a file inside a registered model's directory. */
export function modelFileUrl(avatarId: string, path: string): string {
  return `${root()}/api/v1/live2d/avatars/${avatarId}/model/${path}`;
}

/** fetch with the configured credential attached. */
export function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (endpoint.token) headers.set("X-V1-Token", endpoint.token);
  return fetch(input, { ...init, headers });
}

export const SHADER_PATH = () => runtimeUrl("Shaders/WebGL/");

let corePromise: Promise<void> | null = null;

/**
 * Load the Cubism core.
 *
 * It is an emscripten module, so it cannot go through the bundler -- it has
 * to be a classic script that publishes a global. It must be ready BEFORE
 * any framework module is imported: Framework/src/model/cubismmodel.ts
 * dereferences Live2DCubismCore at module-evaluation time, not at
 * model-load time. That ordering is why stage.ts dynamically imports
 * model.ts after this resolves.
 *
 * Fetched as text and injected from a Blob URL rather than pointed at
 * directly: a bare `<script src>` cannot carry the X-V1-Token header, and
 * routing around that would make the one client the token exists for unable
 * to load its own runtime.
 */
export function loadCore(): Promise<void> {
  corePromise ??= (async () => {
    if ((window as unknown as { Live2DCubismCore?: unknown }).Live2DCubismCore) return;
    const resp = await authFetch(runtimeUrl("Core/live2dcubismcore.js"));
    if (!resp.ok) throw new Error(`failed to load core: ${resp.status}`);
    const url = URL.createObjectURL(new Blob([await resp.text()], { type: "application/javascript" }));
    try {
      await new Promise<void>((resolve, reject) => {
        const el = document.createElement("script");
        el.src = url;
        el.onload = () => resolve();
        el.onerror = () => reject(new Error("core script failed to execute"));
        document.head.appendChild(el);
      });
    } finally {
      URL.revokeObjectURL(url);
    }
  })();
  return corePromise;
}