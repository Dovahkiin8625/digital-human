import path from "node:path";
import { defineConfig, configDefaults } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The pet is loaded by Electron from file://, where an absolute
  // "/assets/..." resolves to "file:///assets/..." and 404s -- a blank
  // transparent window with no error. Relative asset URLs are the only
  // thing file:// accepts.
  base: "./",
  plugins: [react()],
  resolve: {
    // Vite's default list already covers .tsx. Naming it explicitly is a
    // guard: the Cubism SDK's own Demo lists only .ts/.js because it has no
    // JSX, and copying that verbatim silently breaks every React import.
    extensions: [".ts", ".tsx", ".js", ".jsx", ".json"],
    // The Cubism Framework ships TypeScript source with no built bundle, so
    // Vite compiles it inline. Keep in sync with tsconfig's `paths`.
    alias: {
      "@framework": path.resolve(__dirname, "../data/CubismSdkForWeb-5-r.5/Framework/src"),
    },
  },
  server: {
    // The pet's page is opened from file:// by Electron, where every call
    // to the server is cross-origin. The dev server is the one place the
    // pet is same-origin with the API, so the proxy makes `npm run dev`
    // in a plain browser behave like the real app.
    proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true } },
  },
  test: {
    exclude: [...configDefaults.exclude],
  },
});
