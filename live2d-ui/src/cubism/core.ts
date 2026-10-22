/** Cubism runtime bootstrap.
 *
 *  import.meta.env.BASE_URL is "/live2d/" both under vite dev and in the
 *  FastAPI-mounted build, but hardcoding either spelling breaks the other
 *  mode, so every Cubism URL is derived from it.
 */
const CUBISM = `${import.meta.env.BASE_URL}cubism`;

export const SHADER_PATH = `${CUBISM}/Shaders/WebGL/`;
export const CORE_URL = `${CUBISM}/Core/live2dcubismcore.js`;
export const SAMPLE_MODEL = `${CUBISM}/model/Hiyori/Hiyori.model3.json`;

let corePromise: Promise<void> | null = null;

/** Load the Cubism core.
 *
 *  The core is an emscripten module, so it cannot go through the bundler --
 *  it has to be a classic script that publishes a global. It must be ready
 *  BEFORE any framework module is imported: Framework/src/model/cubismmodel.ts
 *  reads Live2DCubismCore at module-evaluation time, not at model-load time.
 *  That ordering is why the framework is dynamically imported in stage.ts
 *  rather than statically imported here. */
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