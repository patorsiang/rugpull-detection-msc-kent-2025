// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 0.0.0.0 in container
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://backend:8000", // service name, not localhost
        changeOrigin: true,
        // DO NOT rewrite
      },
    },
  },
});
