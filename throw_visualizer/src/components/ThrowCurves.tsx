import { useMemo } from 'react'
import * as THREE from 'three'
import type { ThrowRecord } from '../types'
import { extendCurveToTable, gamePointsToThree } from '../coordinates'
import { THROW_POINT_COLOR, throwCurveColor } from '../colors'

const TUBE_RADIUS = 0.01
const POINT_RADIUS = 0.014

interface ThrowCurvesProps {
  throws: ThrowRecord[]
  selectedThrowId: number | null
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

function ThrowPoints({ points }: { points: THREE.Vector3[] }) {
  if (points.length === 0) {
    return null
  }

  return (
    <group>
      {points.map((point, index) => (
        <mesh key={index} position={point}>
          <sphereGeometry args={[POINT_RADIUS, 12, 12]} />
          <meshStandardMaterial color={THROW_POINT_COLOR} roughness={0.25} metalness={0.15} />
        </mesh>
      ))}
    </group>
  )
}

export function ThrowCurves({ throws, selectedThrowId }: ThrowCurvesProps) {
  const selectedThrow = selectedThrowId === null
    ? null
    : throws.find((t) => t.id === selectedThrowId) ?? null

  const selectedPoints = useMemo(
    () => (selectedThrow ? gamePointsToThree(selectedThrow.points_3d) : []),
    [selectedThrow],
  )

  return (
    <group>
      {throws.map((t) => {
        const curvePoints =
          t.fitted_curve_3d.length >= 2 ? t.fitted_curve_3d : t.points_3d
        const points = gamePointsToThree(extendCurveToTable(curvePoints))

        if (points.length < 2) {
          return null
        }

        return (
          <ThrowCurve
            key={t.id}
            points={points}
            color={throwCurveColor(t.id, selectedThrowId)}
          />
        )
      })}
      {selectedPoints.length > 0 && <ThrowPoints points={selectedPoints} />}
    </group>
  )
}
