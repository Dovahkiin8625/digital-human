/** Cubism runtime bootstrap.
 *
 *  The Core and the Shaders come from the server, not from this bundle. They
 *  are Live2D-proprietary, so they are never committed and never shipped in
 *  the build output -- the service serves them from ``DH_LIVE2D_SDK_DIR``,
 *  restricted to the Core/ and Shaders/ subtrees.
 *
 *  The framework fetches its own GLSL with a fetch it does not let the
 *  caller configure with headers, and the core is loaded as a classic
 *  script that cannot carry one either -- so nothing here needs a
 *  credential, and nothing sends one.
 */
interface RuntimeEndpoint {
  /** Absolute server root. Empty in the console, which is served from the
   *  same origin as the API. */
  baseUrl?: string;
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
function runtimeUrl(path: string): string {
  return `${root()}/api/v1/live2d/runtime/${path}`;
}

/** Absolute URL for a file inside a registered model's directory. */
export function modelFileUrl(avatarId: string, path: string): string {
  return `${root()}/api/v1/live2d/avatars/${avatarId}/model/${path}`;
}

/** Fetch an asset. */
export function fetchAsset(input: string, init: RequestInit = {}): Promise<Response> {
  return fetch(input, init);
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
 * directly: the core is a CommonJS-style emscripten bundle, and letting
 * the bundler rewrite it would break the module it publishes.
 */
export function loadCore(): Promise<void> {
  corePromise ??= (async () => {
    if ((window as unknown as { Live2DCubismCore?: unknown }).Live2DCubismCore) return;
    const resp = await fetchAsset(runtimeUrl("Core/live2dcubismcore.js"));
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
