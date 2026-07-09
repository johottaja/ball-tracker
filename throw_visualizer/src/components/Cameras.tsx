import { useMemo } from 'react'
import * as THREE from 'three'
import type { CameraLayoutStats } from '../types'
import { FLOOR_HEIGHT_M } from './Table'
import { gameToThree, opticalAxisFromAngles } from '../coordinates'

const CAMERA_COLORS: Record<string, string> = {
  left: '#c62828',
  right: '#1565c0',
}

const FLOOR_Y = -FLOOR_HEIGHT_M
const TRIPOD_FOOT_RADIUS_M = 0.3
const BODY_WIDTH = 0.14
const BODY_HEIGHT = 0.09
const BODY_DEPTH = 0.07
const LENS_RADIUS = 0.042
const LENS_LENGTH = 0.075
const LEG_RADIUS = 0.011

interface TripodLegProps {
  from: THREE.Vector3
  to: THREE.Vector3
}

function TripodLeg({ from, to }: TripodLegProps) {
  const { position, quaternion, length } = useMemo(() => {
    const direction = to.clone().sub(from)
    const len = direction.length()
    const midpoint = from.clone().add(to).multiplyScalar(0.5)
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.normalize(),
    )
    return { position: midpoint, quaternion, length: len }
  }, [from, to])

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[LEG_RADIUS, LEG_RADIUS * 0.85, length, 8]} />
      <meshStandardMaterial color="#1a1a1a" roughness={0.8} metalness={0.15} />
    </mesh>
  )
}

interface VintageCameraProps {
  layout: CameraLayoutStats
}

function VintageCamera({ layout }: VintageCameraProps) {
  const position = useMemo(
    () => gameToThree(layout.center[0], layout.center[1], layout.center[2]),
    [layout.center],
  )

  const forward = useMemo(() => {
    if (layout.fov_left_xy && layout.fov_right_xy) {
      const [leftX, leftY] = layout.fov_left_xy
      const [rightX, rightY] = layout.fov_right_xy
      const tableTarget = gameToThree((leftX + rightX) / 2, (leftY + rightY) / 2, 0)
      return tableTarget.sub(position).normalize()
    }
    return opticalAxisFromAngles(layout.yaw_deg, layout.pitch_deg)
  }, [layout, position])

  const orientation = useMemo(
    () =>
      new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, -1),
        forward,
      ),
    [forward],
  )

  const footPoints = useMemo(() => {
    const feet: THREE.Vector3[] = []
    for (let i = 0; i < 3; i++) {
      const angle = (i / 3) * Math.PI * 2 + Math.PI / 6
      feet.push(
        new THREE.Vector3(
          Math.cos(angle) * TRIPOD_FOOT_RADIUS_M,
          FLOOR_Y - position.y,
          Math.sin(angle) * TRIPOD_FOOT_RADIUS_M,
        ),
      )
    }
    return feet
  }, [position])

  const accent = CAMERA_COLORS[layout.name] ?? '#555555'

  return (
    <group position={position}>
      {footPoints.map((foot, index) => (
        <TripodLeg key={index} from={new THREE.Vector3()} to={foot} />
      ))}

      <group quaternion={orientation}>
        <mesh position={[0, 0.012, BODY_DEPTH * 0.35]}>
          <boxGeometry args={[BODY_WIDTH, BODY_HEIGHT, BODY_DEPTH]} />
          <meshStandardMaterial color={accent} roughness={0.55} metalness={0.2} />
        </mesh>

        <mesh position={[0, 0.028, -BODY_DEPTH * 0.15]}>
          <boxGeometry args={[BODY_WIDTH * 0.35, BODY_HEIGHT * 0.22, BODY_DEPTH * 0.45]} />
          <meshStandardMaterial color="#111111" roughness={0.7} metalness={0.1} />
        </mesh>

        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.01, -BODY_DEPTH * 0.5 - LENS_LENGTH / 2]}>
          <cylinderGeometry args={[LENS_RADIUS, LENS_RADIUS * 1.08, LENS_LENGTH, 20]} />
          <meshStandardMaterial color="#0d0d0d" roughness={0.35} metalness={0.45} />
        </mesh>

        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.01, -BODY_DEPTH * 0.5 - LENS_LENGTH - 0.012]}>
          <cylinderGeometry args={[LENS_RADIUS * 0.72, LENS_RADIUS * 0.72, 0.018, 20]} />
          <meshStandardMaterial
            color="#4fc3f7"
            roughness={0.15}
            metalness={0.6}
            transparent
            opacity={0.85}
          />
        </mesh>

        <mesh position={[0, -BODY_HEIGHT * 0.42, BODY_DEPTH * 0.42]}>
          <sphereGeometry args={[0.018, 10, 10]} />
          <meshStandardMaterial color="#222222" roughness={0.6} metalness={0.35} />
        </mesh>
      </group>
    </group>
  )
}

interface CamerasProps {
  cameras: CameraLayoutStats[]
}

export function Cameras({ cameras }: CamerasProps) {
  return (
    <group>
      {cameras.map((camera) => (
        <VintageCamera key={camera.name} layout={camera} />
      ))}
    </group>
  )
}
