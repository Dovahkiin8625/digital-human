import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The SPA is mounted by FastAPI at /live2d, so hashed asset URLs must be
  // emitted under that prefix or the browser 404s them -- same reason
  // admin-ui sets base to "/admin/".
  base: "/live2d/",
  plugins: [react()],
  resolve: {
    // The SDK Demo lists only .ts/.js because it has no JSX. This app is
    // React, so the JSX extensions have to be resolvable or `import App from
    // "./App"` cannot find App.tsx.
    extensions: [".ts", ".tsx", ".js", ".jsx", ".json"],
    // The Cubism Framework ships TypeScript source with no built bundle, so
    // Vite compiles it inline. This is the wiring the SDK's own Demo uses.
    // The SDK is proprietary and lives outside the repo tree -- see the
    // ledger ruling about relocating it out of data/ before shipping.
    alias: {
      "@framework": path.resolve(__dirname, "../data/CubismSdkForWeb-5-r.5/Framework/src"),
    },
  },
  server: {
    // `ws: true` is required for the /api proxy to carry WebSocket upgrades.
    proxy: { "/api": { target: "http://127.0.0.1:8000", ws: true } },
  },
});