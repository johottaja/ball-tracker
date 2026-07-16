export const DEFAULT_THROW_COLOR = 'hsl(0, 78%, 52%)'
export const DIMMED_THROW_COLOR = 'hsl(0, 50%, 30%)'
export const HIGHLIGHTED_THROW_COLOR = 'hsl(0, 88%, 68%)'
export const THROW_POINT_COLOR = 'hsl(48, 95%, 62%)'
export const LEFT_THROW_COLOR = 'hsl(210, 78%, 52%)'
export const LEFT_DIMMED_THROW_COLOR = 'hsl(210, 50%, 30%)'
export const LEFT_HIGHLIGHTED_THROW_COLOR = 'hsl(210, 88%, 68%)'

export function throwCurveColor(
  throwId: number,
  selectedThrowId: number | null,
  throwerSide: 'left' | 'right' = 'right',
): string {
  const isLeftThrow = throwerSide === 'left'
  if (selectedThrowId === null) {
    return isLeftThrow ? LEFT_THROW_COLOR : DEFAULT_THROW_COLOR
  }
  if (throwId === selectedThrowId) {
    return isLeftThrow ? LEFT_HIGHLIGHTED_THROW_COLOR : HIGHLIGHTED_THROW_COLOR
  }
  return isLeftThrow ? LEFT_DIMMED_THROW_COLOR : DIMMED_THROW_COLOR
}
