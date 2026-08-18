import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import seo from './vite-plugin-seo'

// Backend target for the dev proxy. Defaults to the docker-compose service
// name; set VITE_PROXY_TARGET=http://localhost:8000 to run the dev server on
// the host against a backend reachable at localhost (no CORS, same-origin).
const backend = process.env.VITE_PROXY_TARGET || 'http://backend:8000'
const renderer = process.env.VITE_RENDER_TARGET || 'http://renderer:3100'

// Vite 4.5.6+ blocks unknown Host headers (DNS rebinding). localhost / IPs
// stay allowed. The official compose `dev` stage is what self-host (Dokploy
// included) runs, so public hostnames must be listed here or passed in
// VITE_ALLOWED_HOSTS (comma-separated). `true` or `*` allows any host.
const DEFAULT_ALLOWED_HOSTS = [
  'openshorts.app',
  'www.openshorts.app',
  'openshorts.julioakamine.com',
]

function resolveAllowedHosts(raw = process.env.VITE_ALLOWED_HOSTS) {
  const extra = (raw || '').trim()
  if (extra === 'true' || extra === '*') return true
  const extras = extra.split(',').map((host) => host.trim()).filter(Boolean)
  return [...new Set([...DEFAULT_ALLOWED_HOSTS, ...extras])]
}

// https://vitejs.dev/config/
export default defineConfig({
  // seo() runs on build only. It injects the crawler-visible homepage content
  // into #root and emits the static /alternatives pages, sitemap.xml and
  // llms.txt. See vite-plugin-seo.js.
  plugins: [react(), seo()],
  server: {
    allowedHosts: resolveAllowedHosts(),
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/videos': { target: backend, changeOrigin: true },
      '/thumbnails': { target: backend, changeOrigin: true },
      '/gallery': { target: backend, changeOrigin: true },
      '/video': { target: backend, changeOrigin: true },
      '/render': { target: renderer, changeOrigin: true },
    }
  }
})
