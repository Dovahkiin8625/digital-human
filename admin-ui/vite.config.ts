import path from "node:path";
import { defineConfig, configDefaults } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  // The console is mounted by FastAPI at /admin, so hashed asset URLs must
  // be emitted under that prefix or the browser 404s them.
  //
  // The Electron pet loads the same bundle from file://, where an absolute
  // "/admin/assets/..." resolves to "file:///admin/assets/..." and silently
  // 404s -- a blank transparent window with no error. Its build therefore
  // uses relative asset URLs.
  base: mode === "pet" ? "./" : "/admin/",
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
    // `ws: true` is required: the debug console opens WebSocket upgrades under
    // /api (/api/v1/ws/signal/... and .../audio/...). Without it the dev proxy
    // forwards only plain HTTP, the upgrade is never answered, and the browser's
    // WebSocket sits in CONNECTING forever -- no error, no console message, and
    // the Debug page spins on 开始调试 with the 合成并驱动 button stuck disabled.
    proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true } },
  },
  test: {
    // reentryGuard.test.ts is deliberately node:test, not vitest: Node
    // strips its TS types directly, and tests/test_admin_ui.py shells out to
    // `node --test` on this exact file to prove the guard's behaviour rather
    // than grep its source. Vitest collecting it would report "No test suite
    // found" and fail.
    exclude: [...configDefaults.exclude, "src/pages/reentryGuard.test.ts"],
  },
}));
