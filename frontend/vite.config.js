import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // ── Development server ──────────────────────────────────────────────────────
  server: {
    port: 5173,
    proxy: {
      // Proxy /api/* and /static/* → Flask backend during local development.
      // In production these requests go directly to the deployed backend URL
      // configured via VITE_API_BASE_URL environment variable.
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },

  // ── Production build ────────────────────────────────────────────────────────
  build: {
    outDir: 'dist',
  },
})
