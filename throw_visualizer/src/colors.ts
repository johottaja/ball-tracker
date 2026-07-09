export function throwColor(index: number, total: number): string {
  if (total <= 1) {
    return 'hsl(0, 78%, 52%)'
  }
  const lightness = 32 + (index / (total - 1)) * 38
  const saturation = 68 + (index / (total - 1)) * 14
  return `hsl(0, ${saturation}%, ${lightness}%)`
}
