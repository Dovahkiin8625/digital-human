import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The SPA is mounted by FastAPI at /admin, so hashed asset URLs must be
  // emitted under that prefix (/admin/assets/...) or the browser 404s them.
  base: "/admin/",
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
});
