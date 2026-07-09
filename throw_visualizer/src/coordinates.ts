import * as THREE from 'three'

/** Game coords: X = length, Y = width, Z = up. Three.js uses Y-up. */
export function gameToThree(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, y)
}

export function gamePointsToThree(
  points: { x: number; y: number; z: number }[],
): THREE.Vector3[] {
  return points.map((p) => gameToThree(p.x, p.y, p.z))
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
