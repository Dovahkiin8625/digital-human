import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // The SPA is mounted by FastAPI at /admin, so hashed asset URLs must be
  // emitted under that prefix (/admin/assets/...) or the browser 404s them.
  base: "/admin/",
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
