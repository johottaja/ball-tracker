import type { GameSession } from './types'

export interface GameListEntry {
  filename: string
  label: string
  modifiedAt: string
}

export const DEFAULT_TABLE_LENGTH_M = 1.5
export const DEFAULT_TABLE_WIDTH_M = 0.6

export async function fetchGameList(): Promise<GameListEntry[]> {
  const response = await fetch('/api/games')
  if (!response.ok) {
    throw new Error(`Failed to list games (${response.status})`)
  }
  const payload = (await response.json()) as { games: GameListEntry[] }
  return payload.games
}

export async function fetchGame(filename: string): Promise<GameSession> {
  const response = await fetch(`/api/games/${encodeURIComponent(filename)}`)
  if (!response.ok) {
    throw new Error(`Failed to load game (${response.status})`)
  }
  return (await response.json()) as GameSession
}

export function tableDimensions(game: GameSession): {
  lengthM: number
  widthM: number
} {
  return {
    lengthM: game.calibration?.table_length_m ?? DEFAULT_TABLE_LENGTH_M,
    widthM: game.calibration?.table_width_m ?? DEFAULT_TABLE_WIDTH_M,
  }
}
