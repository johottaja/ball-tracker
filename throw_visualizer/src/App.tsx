import { useCallback, useEffect, useState } from 'react'
import { Scene } from './components/Scene'
import { fetchGame, fetchGameList } from './games'
import type { GameListEntry } from './games'
import type { GameSession } from './types'

function App() {
  const [gameList, setGameList] = useState<GameListEntry[]>([])
  const [selectedFilename, setSelectedFilename] = useState('')
  const [game, setGame] = useState<GameSession | null>(null)
  const [uploadedGame, setUploadedGame] = useState<GameSession | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [gameError, setGameError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [selectedThrowId, setSelectedThrowId] = useState<number | null>(null)

  const refreshGameList = useCallback(async (selectLatest = false) => {
    setRefreshing(true)
    setListError(null)
    try {
      const games = await fetchGameList()
      setGameList(games)
      setSelectedFilename((current) => {
        if (selectLatest && games.length > 0) {
          return games[0].filename
        }
        if (current && games.some((entry) => entry.filename === current)) {
          return current
        }
        return games[0]?.filename ?? ''
      })
    } catch (error) {
      setListError(error instanceof Error ? error.message : 'Failed to load games')
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void refreshGameList()
  }, [refreshGameList])

  useEffect(() => {
    const onFocus = () => {
      void refreshGameList()
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [refreshGameList])

  useEffect(() => {
    if (uploadedGame) {
      return
    }
    if (!selectedFilename) {
      setGame(null)
      return
    }

    let cancelled = false
    setGameError(null)

    void fetchGame(selectedFilename)
      .then((loaded) => {
        if (!cancelled) {
          setGame(loaded)
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setGame(null)
          setGameError(error instanceof Error ? error.message : 'Failed to load game')
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedFilename, uploadedGame])

  const activeGame = uploadedGame ?? game

  useEffect(() => {
    setSelectedThrowId(null)
  }, [activeGame])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    try {
      const text = await file.text()
      setUploadedGame(JSON.parse(text) as GameSession)
      setGameError(null)
    } catch {
      setUploadedGame(null)
      setGameError('Invalid game JSON file')
    }
    event.target.value = ''
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950 text-zinc-100">
      <header className="flex shrink-0 flex-wrap items-center gap-4 border-b border-zinc-800 px-4 py-3">
        <h1 className="text-lg font-semibold tracking-tight">Throw visualizer</h1>

        {gameList.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-zinc-400">
            Game
            <select
              className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
              value={selectedFilename}
              onChange={(e) => {
                setUploadedGame(null)
                setSelectedFilename(e.target.value)
              }}
              disabled={uploadedGame !== null}
            >
              {gameList.map((entry) => (
                <option key={entry.filename} value={entry.filename}>
                  {entry.label}
                </option>
              ))}
            </select>
          </label>
        )}

        <button
          type="button"
          className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
          onClick={() => void refreshGameList(true)}
          disabled={refreshing}
        >
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>

        <label className="cursor-pointer rounded-md border border-zinc-700 bg-zinc-900 px-3 py-1 text-sm text-zinc-300 hover:bg-zinc-800">
          Load JSON
          <input
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={handleFileUpload}
          />
        </label>

        {uploadedGame && (
          <button
            type="button"
            className="text-sm text-zinc-500 hover:text-zinc-300"
            onClick={() => setUploadedGame(null)}
          >
            Clear upload
          </button>
        )}

        {activeGame && activeGame.throws.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-zinc-400">
            Throw
            <select
              className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
              value={selectedThrowId ?? ''}
              onChange={(e) => {
                const value = e.target.value
                setSelectedThrowId(value === '' ? null : Number(value))
              }}
            >
              <option value="">All throws</option>
              {activeGame.throws.map((t) => (
                <option key={t.id} value={t.id}>
                  Throw {t.id}
                  {t.speed_m_s !== null ? ` · ${t.speed_m_s.toFixed(1)} m/s` : ''}
                </option>
              ))}
            </select>
          </label>
        )}

        {activeGame && (
          <span className="ml-auto text-sm text-zinc-500">
            Table {activeGame.calibration?.table_length_m ?? 1.5} ×{' '}
            {activeGame.calibration?.table_width_m ?? 0.6} m · {activeGame.throws.length} throws
          </span>
        )}
      </header>

      <main className="relative min-h-0 flex-1">
        {listError && (
          <div className="absolute left-4 top-4 z-10 rounded-md border border-red-900 bg-red-950/80 px-3 py-2 text-sm text-red-200">
            {listError}
          </div>
        )}
        {gameError && (
          <div className="absolute left-4 top-14 z-10 rounded-md border border-red-900 bg-red-950/80 px-3 py-2 text-sm text-red-200">
            {gameError}
          </div>
        )}
        {activeGame ? (
          <Scene game={activeGame} selectedThrowId={selectedThrowId} />
        ) : (
          <div className="flex h-full items-center justify-center text-zinc-500">
            No game JSON found. Process a game in game_tracker, then hit Refresh.
          </div>
        )}
      </main>
    </div>
  )
}

export default App
