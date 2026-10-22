/**
 * Stage the Cubism runtime into public/ so Vite can serve it.
 *
 * Everything copied here is Live2D-proprietary or Live2D sample content,
 * so the destination is gitignored -- this repo must never carry it.
 * Mirrors the SDK's own Samples/TypeScript/Demo/copy_resources.js.
 *
 * Override the SDK location with CUBISM_SDK_DIR.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const sdk = process.env.CUBISM_SDK_DIR ?? path.resolve(root, "../data/CubismSdkForWeb-5-r.5");
const dest = path.join(root, "public", "cubism");

if (!fs.existsSync(path.join(sdk, "Core"))) {
  console.error(`Cubism SDK not found at ${sdk}\nSet CUBISM_SDK_DIR to the SDK root.`);
  process.exit(1);
}

fs.rmSync(dest, { recursive: true, force: true });
fs.mkdirSync(dest, { recursive: true });

// The core is loaded as a classic <script> in index.html; it is an emscripten
// module and cannot be imported through the bundler.
fs.cpSync(path.join(sdk, "Core"), path.join(dest, "Core"), { recursive: true });

// The framework fetches its GLSL at runtime by path, so these have to be
// served rather than bundled.
fs.cpSync(path.join(sdk, "Framework", "Shaders"), path.join(dest, "Shaders"), {
  recursive: true,
});

// One sample model, so the debug stage has something to load before the
// server-side model hosting exists.
fs.cpSync(path.join(sdk, "Samples", "Resources", "Hiyori"), path.join(dest, "model", "Hiyori"), {
  recursive: true,
});

console.log(`staged Cubism into ${dest}`);