/** Cubism runtime bootstrap.
 *
 *  The Core and the Shaders come from the server, not from this bundle. They
 *  are Live2D-proprietary, so they are never committed and never shipped in
 *  the build output -- the service serves them from ``DH_LIVE2D_SDK_DIR``,
 *  restricted to the Core/ and Shaders/ subtrees.
 *
 *  That endpoint is deliberately not token-gated: the core loads through a
 *  ``<script>`` tag, which cannot carry a custom header, and the framework
 *  fetches its own GLSL with a fetch the caller cannot configure with one.
 *  Model files, which are operator-supplied, are gated separately.
 */
const RUNTIME = "/api/v1/live2d/runtime";

export const SHADER_PATH = `${RUNTIME}/Shaders/WebGL/`;
export const CORE_URL = `${RUNTIME}/Core/live2dcubismcore.js`;

let corePromise: Promise<void> | null = null;

/** Load the Cubism core.
 *
 *  It is an emscripten module, so it cannot go through the bundler -- it has
 *  to be a classic script that publishes a global. It must be ready BEFORE
 *  any framework module is imported: Framework/src/model/cubismmodel.ts
 *  dereferences Live2DCubismCore at module-evaluation time, not at
 *  model-load time. That ordering is why stage.ts dynamically imports
 *  model.ts after this resolves. */
export function loadCore(): Promise<void> {
  corePromise ??= new Promise<void>((resolve, reject) => {
    if ((window as unknown as { Live2DCubismCore?: unknown }).Live2DCubismCore) {
      resolve();
      return;
    }
    const el = document.createElement("script");
    el.src = CORE_URL;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(`failed to load ${CORE_URL}`));
    document.head.appendChild(el);
  });
  return corePromise;
}