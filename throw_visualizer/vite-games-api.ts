import fs from 'node:fs'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin, PreviewServer, ViteDevServer } from 'vite'

const GAME_FILENAME_RE = /^game-[\w-]+\.json$/

function gamesDir(root: string): string {
  return path.resolve(root, '../game_tracker/games')
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(body))
}

function handleGamesRequest(
  req: IncomingMessage,
  res: ServerResponse,
  root: string,
): void {
  const dir = gamesDir(root)
  const url = req.url ?? '/'
  const pathname = url.split('?')[0] ?? '/'

  if (req.method !== 'GET') {
    sendJson(res, 405, { error: 'Method not allowed' })
    return
  }

  if (pathname === '/' || pathname === '') {
    if (!fs.existsSync(dir)) {
      sendJson(res, 200, { games: [] })
      return
    }

    const games = fs
      .readdirSync(dir)
      .filter((name) => GAME_FILENAME_RE.test(name))
      .map((filename) => {
        const stat = fs.statSync(path.join(dir, filename))
        return {
          filename,
          label: filename.replace(/^game-/, '').replace(/\.json$/, ''),
          modifiedAt: stat.mtime.toISOString(),
        }
      })
      .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt))

    sendJson(res, 200, { games })
    return
  }

  const filename = decodeURIComponent(pathname.slice(1))
  if (!GAME_FILENAME_RE.test(filename)) {
    sendJson(res, 400, { error: 'Invalid game filename' })
    return
  }

  const filePath = path.join(dir, filename)
  if (!fs.existsSync(filePath)) {
    sendJson(res, 404, { error: 'Game not found' })
    return
  }

  res.statusCode = 200
  res.setHeader('Content-Type', 'application/json')
  res.end(fs.readFileSync(filePath, 'utf8'))
}

function attachMiddleware(server: ViteDevServer | PreviewServer, root: string): void {
  server.middlewares.use('/api/games', (req, res) => {
    handleGamesRequest(req, res, root)
  })
}

export function gamesApiPlugin(): Plugin {
  return {
    name: 'games-api',
    configureServer(server) {
      attachMiddleware(server, server.config.root)
    },
    configurePreviewServer(server) {
      attachMiddleware(server, server.config.root)
    },
  }
}
