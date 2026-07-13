import * as THREE from 'three'

export interface GamePoint3D {
  x: number
  y: number
  z: number
}

const TABLE_Z = 0
const TABLE_Z_EPS = 1e-5

/** Game coords: X = length, Y = width, Z = up. Three.js uses Y-up. */
export function gameToThree(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, y)
}

export function gamePointsToThree(points: GamePoint3D[]): THREE.Vector3[] {
  return points.map((p) => gameToThree(p.x, p.y, p.z))
}

function evalQuadratic([a, b, c]: [number, number, number], t: number): number {
  return ((a * t) + b) * t + c
}

function solve3x3(matrix: number[][], values: number[]): [number, number, number] | null {
  const m = matrix.map((row, rowIndex) => [...row, values[rowIndex]!])

  for (let col = 0; col < 3; col++) {
    let pivotRow = col
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row]![col]!) > Math.abs(m[pivotRow]![col]!)) {
        pivotRow = row
      }
    }

    if (Math.abs(m[pivotRow]![col]!) < 1e-12) {
      return null
    }

    if (pivotRow !== col) {
      const swap = m[col]!
      m[col] = m[pivotRow]!
      m[pivotRow] = swap
    }

    for (let row = col + 1; row < 3; row++) {
      const factor = m[row]![col]! / m[col]![col]!
      for (let j = col; j < 4; j++) {
        m[row]![j]! -= factor * m[col]![j]!
      }
    }
  }

  const solution: number[] = [0, 0, 0]
  for (let row = 2; row >= 0; row--) {
    let value = m[row]![3]!
    for (let col = row + 1; col < 3; col++) {
      value -= m[row]![col]! * solution[col]!
    }
    solution[row] = value / m[row]![row]!
  }

  return [solution[0]!, solution[1]!, solution[2]!]
}

function quadraticThroughSamples(
  samples: { t: number; value: number }[],
): [number, number, number] | null {
  if (samples.length < 3) {
    return null
  }

  const [a, b, c] = [samples[0]!, samples[Math.floor(samples.length / 2)]!, samples[samples.length - 1]!]
  return solve3x3(
    [
      [a.t * a.t, a.t, 1],
      [b.t * b.t, b.t, 1],
      [c.t * c.t, c.t, 1],
    ],
    [a.value, b.value, c.value],
  )
}

function trajectoryQuadraticCoeffs(
  points: GamePoint3D[],
): { x: [number, number, number]; y: [number, number, number]; z: [number, number, number] } | null {
  if (points.length < 3) {
    return null
  }

  const lastIndex = points.length - 1
  const sampleIndices = [0, Math.floor(lastIndex / 2), lastIndex]
  const samples = sampleIndices.map((index) => ({
    t: index / lastIndex,
    x: points[index]!.x,
    y: points[index]!.y,
    z: points[index]!.z,
  }))

  const x = quadraticThroughSamples(samples.map((sample) => ({ t: sample.t, value: sample.x })))
  const y = quadraticThroughSamples(samples.map((sample) => ({ t: sample.t, value: sample.y })))
  const z = quadraticThroughSamples(samples.map((sample) => ({ t: sample.t, value: sample.z })))

  if (!x || !y || !z) {
    return null
  }

  return { x, y, z }
}

function solveQuadraticRoots(a: number, b: number, c: number): number[] {
  if (Math.abs(a) < 1e-12) {
    if (Math.abs(b) < 1e-12) {
      return []
    }
    return [-c / b]
  }

  const discriminant = b * b - 4 * a * c
  if (discriminant < 0) {
    return []
  }

  const sqrtDiscriminant = Math.sqrt(discriminant)
  const denom = 2 * a
  return [(-b - sqrtDiscriminant) / denom, (-b + sqrtDiscriminant) / denom]
}

function landingTimeBeyondEnd(zCoeffs: [number, number, number]): number | null {
  const endZ = evalQuadratic(zCoeffs, 1)
  if (endZ <= TABLE_Z + TABLE_Z_EPS) {
    return null
  }

  const endSlope = 2 * zCoeffs[0] + zCoeffs[1]
  if (endSlope >= -TABLE_Z_EPS) {
    return null
  }

  const roots = solveQuadraticRoots(...zCoeffs)
    .filter((t) => t > 1 + 1e-6)
    .sort((left, right) => left - right)

  return roots[0] ?? null
}

function sampleQuadraticCurve(
  coeffs: { x: [number, number, number]; y: [number, number, number]; z: [number, number, number] },
  endT: number,
  sampleCount: number,
): GamePoint3D[] {
  const samples: GamePoint3D[] = []
  const count = Math.max(3, sampleCount)

  for (let index = 0; index < count; index++) {
    const t = (index / (count - 1)) * endT
    samples.push({
      x: evalQuadratic(coeffs.x, t),
      y: evalQuadratic(coeffs.y, t),
      z: Math.max(TABLE_Z, evalQuadratic(coeffs.z, t)),
    })
  }

  samples[samples.length - 1]!.z = TABLE_Z
  return samples
}

/** Extend a descending trajectory forward until it reaches the table (z = 0). */
export function extendCurveToTable(points: GamePoint3D[]): GamePoint3D[] {
  if (points.length < 3) {
    return points
  }

  const last = points[points.length - 1]!
  if (last.z <= TABLE_Z + TABLE_Z_EPS) {
    return points
  }

  const prev = points[points.length - 2]!
  if (last.z >= prev.z - TABLE_Z_EPS) {
    return points
  }

  const coeffs = trajectoryQuadraticCoeffs(points)
  if (!coeffs) {
    return points
  }

  const landingT = landingTimeBeyondEnd(coeffs.z)
  if (landingT === null) {
    return points
  }

  const sampleCount = Math.max(points.length, Math.ceil(points.length * landingT))
  return sampleQuadraticCurve(coeffs, landingT, sampleCount)
}

/** Unit optical-axis direction in game coordinates from layout yaw/pitch. */
export function opticalAxisFromAngles(yawDeg: number, pitchDeg: number): THREE.Vector3 {
  const yaw = THREE.MathUtils.degToRad(yawDeg)
  const pitch = THREE.MathUtils.degToRad(pitchDeg)
  const gx = Math.cos(pitch) * Math.cos(yaw)
  const gy = Math.cos(pitch) * Math.sin(yaw)
  const gz = Math.sin(pitch)
  return gameToThree(gx, gy, gz).normalize()
}
