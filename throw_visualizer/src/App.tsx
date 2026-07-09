import { useMemo, useState } from 'react'
import { Scene } from './components/Scene'
import { availableGames } from './games'
import type { GameSession } from './types'

function App() {
  const [selectedFilename, setSelectedFilename] = useState(
    () => availableGames[0]?.filename ?? '',
  )
  const [uploadedGame, setUploadedGame] = useState<GameSession | null>(null)

  const bundledGame = useMemo(
    () => availableGames.find((g) => g.filename === selectedFilename)?.data ?? null,
    [selectedFilename],
  )

  const game = uploadedGame ?? bundledGame

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    try {
      const text = await file.text()
      setUploadedGame(JSON.parse(text) as GameSession)
    } catch {
      setUploadedGame(null)
    }
    event.target.value = ''
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950 text-zinc-100">
      <header className="flex shrink-0 flex-wrap items-center gap-4 border-b border-zinc-800 px-4 py-3">
        <h1 className="text-lg font-semibold tracking-tight">Throw visualizer</h1>

        {availableGames.length > 0 && (
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
              {availableGames.map((entry) => (
                <option key={entry.filename} value={entry.filename}>
                  {entry.label} ({entry.data.throws.length} throws)
                </option>
              ))}
            </select>
          </label>
        )}

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

        {game && (
          <span className="ml-auto text-sm text-zinc-500">
            Table {game.calibration?.table_length_m ?? 1.5} ×{' '}
            {game.calibration?.table_width_m ?? 0.6} m · {game.throws.length} throws
          </span>
        )}
      </header>

      <main className="relative min-h-0 flex-1">
        {game ? (
          <Scene game={game} />
        ) : (
          <div className="flex h-full items-center justify-center text-zinc-500">
            No game JSON found. Process a game in game_tracker or load a file.
          </div>
        )}
      </main>
    </div>
  )
}

export default App
