export interface CurvePoint3D {
  x: number
  y: number
  z: number
}

export type CurveFitMode = 'quadratic' | 'ballistic'

export interface ThrowRecord {
  id: number
  start_frame: number
  end_frame: number
  points_3d: CurvePoint3D[]
  fitted_curve_3d: CurvePoint3D[]
  speed_m_s: number | null
  ballistic_curve_3d?: CurvePoint3D[]
  ballistic_speed_m_s?: number | null
  thrower_side?: 'left' | 'right'
}

export interface CameraLayoutStats {
  name: string
  center: [number, number, number]
  xy_distance_m: number
  z_m: number
  yaw_deg: number
  pitch_deg: number
  horizontal_fov_deg: number
  fov_left_xy?: [number, number]
  fov_right_xy?: [number, number]
}

export interface GameCalibration {
  table_length_m: number
  table_width_m: number
  layout?: {
    cameras: CameraLayoutStats[]
  }
}

export interface GameSession {
  version: number
  recorded_at: string
  fps: number
  frame_count: number
  coordinate_system?: Record<string, string>
  calibration?: GameCalibration
  throws: ThrowRecord[]
}
