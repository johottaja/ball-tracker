export const DEFAULT_THROW_COLOR = 'hsl(0, 78%, 52%)'
export const DIMMED_THROW_COLOR = 'hsl(0, 50%, 30%)'
export const HIGHLIGHTED_THROW_COLOR = 'hsl(0, 88%, 68%)'
export const THROW_POINT_COLOR = 'hsl(48, 95%, 62%)'

export function throwCurveColor(throwId: number, selectedThrowId: number | null): string {
  if (selectedThrowId === null) {
    return DEFAULT_THROW_COLOR
  }
  return throwId === selectedThrowId ? HIGHLIGHTED_THROW_COLOR : DIMMED_THROW_COLOR
}
