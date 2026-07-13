import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { gamesApiPlugin } from './vite-games-api.js'

export default defineConfig({
  plugins: [react(), tailwindcss(), gamesApiPlugin()],
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
