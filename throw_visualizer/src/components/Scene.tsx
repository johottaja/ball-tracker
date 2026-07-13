import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { GameSession } from '../types'
import { tableDimensions } from '../games'
import { Table } from './Table'
import { ThrowCurves } from './ThrowCurves'
import { Cameras } from './Cameras'

interface SceneProps {
  game: GameSession
  selectedThrowId: number | null
}

function SceneContent({ game, selectedThrowId }: SceneProps) {
  const { lengthM, widthM } = tableDimensions(game)

  return (
    <>
      <ambientLight intensity={0.45} />
      <directionalLight position={[4, 6, 3]} intensity={1.1} castShadow />
      <directionalLight position={[-3, 4, -2]} intensity={0.35} />

      <Table lengthM={lengthM} widthM={widthM} />
      {game.calibration?.layout?.cameras && (
        <Cameras cameras={game.calibration.layout.cameras} />
      )}
      <ThrowCurves throws={game.throws} selectedThrowId={selectedThrowId} />

      <OrbitControls
        makeDefault
        target={[0, 0, 0]}
        minDistance={0.5}
        maxDistance={12}
        maxPolarAngle={Math.PI / 2 - 0.05}
      />
    </>
  )
}

export function Scene({ game, selectedThrowId }: SceneProps) {
  return (
    <Canvas
      camera={{ position: [2.2, 1.4, 2.2], fov: 50, near: 0.05, far: 50 }}
      shadows
      className="h-full w-full"
    >
      <color attach="background" args={['#0f1115']} />
      <fog attach="fog" args={['#0f1115', 6, 18]} />
      <SceneContent game={game} selectedThrowId={selectedThrowId} />
    </Canvas>
  )
}
