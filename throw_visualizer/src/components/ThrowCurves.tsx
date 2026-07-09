import { useMemo } from 'react'
import * as THREE from 'three'
import type { ThrowRecord } from '../types'
import { gamePointsToThree } from '../coordinates'
import { throwColor } from '../colors'

const TUBE_RADIUS = 0.01

interface ThrowCurvesProps {
  throws: ThrowRecord[]
}

function ThrowCurve({ points, color }: { points: THREE.Vector3[]; color: string }) {
  const geometry = useMemo(() => {
    if (points.length < 2) {
      return null
    }
    const curve = new THREE.CatmullRomCurve3(points)
    return new THREE.TubeGeometry(curve, Math.max(points.length * 2, 32), TUBE_RADIUS, 8, false)
  }, [points])

  if (!geometry) {
    return null
  }

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} roughness={0.35} metalness={0.1} />
    </mesh>
  )
}

export function ThrowCurves({ throws }: ThrowCurvesProps) {
  return (
    <group>
      {throws.map((t, index) => {
        const points =
          t.fitted_curve_3d.length >= 2
            ? gamePointsToThree(t.fitted_curve_3d)
            : gamePointsToThree(t.points_3d)

        if (points.length < 2) {
          return null
        }

        return (
          <ThrowCurve
            key={t.id}
            points={points}
            color={throwColor(index, throws.length)}
          />
        )
      })}
    </group>
  )
}
