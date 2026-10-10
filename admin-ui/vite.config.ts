import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The SPA is mounted by FastAPI at /admin, so hashed asset URLs must be
  // emitted under that prefix (/admin/assets/...) or the browser 404s them.
  base: "/admin/",
  plugins: [react()],
  server: {
    // `ws: true` is required: the debug console opens WebSocket upgrades under
    // /api (/api/v1/ws/signal/... and .../audio/...). Without it the dev proxy
    // forwards only plain HTTP, the upgrade is never answered, and the browser's
    // WebSocket sits in CONNECTING forever -- no error, no console message, and
    // the Debug page spins on 开始调试 with the 合成并驱动 button stuck disabled.
    proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true } },
  },
});
