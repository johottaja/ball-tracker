const TABLE_TOP_THICKNESS_M = 0.04
const LEG_RADIUS_M = 0.035
const LEG_INSET_M = 0.08
export const FLOOR_HEIGHT_M = 0.8

interface TableProps {
  lengthM: number
  widthM: number
}

export function Table({ lengthM, widthM }: TableProps) {
  const legHeight = FLOOR_HEIGHT_M - TABLE_TOP_THICKNESS_M
  const halfL = lengthM / 2
  const halfW = widthM / 2

  const legCenters: [number, number][] = [
    [-halfL + LEG_INSET_M, -halfW + LEG_INSET_M],
    [halfL - LEG_INSET_M, -halfW + LEG_INSET_M],
    [-halfL + LEG_INSET_M, halfW - LEG_INSET_M],
    [halfL - LEG_INSET_M, halfW - LEG_INSET_M],
  ]

  return (
    <group>
      <mesh position={[0, -TABLE_TOP_THICKNESS_M / 2, 0]}>
        <boxGeometry args={[lengthM, TABLE_TOP_THICKNESS_M, widthM]} />
        <meshStandardMaterial color="#2d6a4f" roughness={0.65} metalness={0.05} />
      </mesh>

      {legCenters.map(([x, z]) => (
        <mesh key={`${x}-${z}`} position={[x, -FLOOR_HEIGHT_M + legHeight / 2, z]}>
          <cylinderGeometry args={[LEG_RADIUS_M, LEG_RADIUS_M, legHeight, 16]} />
          <meshStandardMaterial color="#1b4332" roughness={0.75} metalness={0.05} />
        </mesh>
      ))}

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -FLOOR_HEIGHT_M, 0]} receiveShadow>
        <planeGeometry args={[8, 8]} />
        <meshStandardMaterial color="#2b2b2b" roughness={0.95} metalness={0} />
      </mesh>
    </group>
  )
}
