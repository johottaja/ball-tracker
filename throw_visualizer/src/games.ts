import type { GameSession } from './types'

export interface GameEntry {
  filename: string
  label: string
  data: GameSession
}

const modules = import.meta.glob<GameSession>(
  '../../game_tracker/games/*.json',
  { eager: true, import: 'default' },
)

export const availableGames: GameEntry[] = Object.entries(modules)
  .map(([path, data]) => {
    const filename = path.split('/').pop() ?? path
    return {
      filename,
      label: filename.replace(/^game-/, '').replace(/\.json$/, ''),
      data,
    }
  })
  .sort((a, b) => b.label.localeCompare(a.label))

export const DEFAULT_TABLE_LENGTH_M = 1.5
export const DEFAULT_TABLE_WIDTH_M = 0.6

export function tableDimensions(game: GameSession): {
  lengthM: number
  widthM: number
} {
  return {
    lengthM: game.calibration?.table_length_m ?? DEFAULT_TABLE_LENGTH_M,
    widthM: game.calibration?.table_width_m ?? DEFAULT_TABLE_WIDTH_M,
  }
}
