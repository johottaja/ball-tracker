from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import MAX_TRIANGULATION_HEIGHT_M, MAX_TRIANGULATION_RESIDUAL_M
from .game_data import CurvePoint3D, Point2D, Point3D, ThrowRecord
from .setup_config import CameraSetup

TABLE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class StereoCameraModel:
    """Cached intrinsics and projection matrices for both cameras."""

    width: int
    height: int
    k: np.ndarray
    p_main: np.ndarray
    p_secondary: np.ndarray
    main_position: np.ndarray
    secondary_position: np.ndarray


def _intrinsic_matrix(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    fov_rad = math.radians(horizontal_fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _look_at_rt(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera R and t for OpenCV (camera looks along +Z)."""
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / forward_norm

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm

    down = np.cross(forward, right)
    r = np.vstack([right, down, forward])
    t = -r @ position
    return r, t


def camera_positions(setup: CameraSetup) -> tuple[np.ndarray, np.ndarray]:
    """Primary at middle of -Y end; secondary at azimuth -90° + angle."""
    main = np.array(
        [0.0, -setup.main_distance_m, setup.main_height_m],
        dtype=np.float64,
    )
    azimuth_rad = math.radians(-90.0 + setup.camera_angle_deg)
    secondary = np.array(
        [
            setup.secondary_distance_m * math.cos(azimuth_rad),
            setup.secondary_distance_m * math.sin(azimuth_rad),
            setup.secondary_height_m,
        ],
        dtype=np.float64,
    )
    return main, secondary


def build_stereo_camera_model(
    setup: CameraSetup,
    *,
    width: int,
    height: int,
) -> StereoCameraModel:
    k = _intrinsic_matrix(width, height, setup.horizontal_fov_deg)
    main_pos, secondary_pos = camera_positions(setup)

    r_main, t_main = _look_at_rt(main_pos, TABLE_CENTER, WORLD_UP)
    r_secondary, t_secondary = _look_at_rt(secondary_pos, TABLE_CENTER, WORLD_UP)

    rt_main = np.hstack([r_main, t_main.reshape(3, 1)])
    rt_secondary = np.hstack([r_secondary, t_secondary.reshape(3, 1)])

    return StereoCameraModel(
        width=width,
        height=height,
        k=k,
        p_main=k @ rt_main,
        p_secondary=k @ rt_secondary,
        main_position=main_pos,
        secondary_position=secondary_pos,
    )


def _triangulate_point(
    p_main: np.ndarray,
    p_secondary: np.ndarray,
    model: StereoCameraModel,
) -> tuple[np.ndarray | None, float]:
    pts_main = np.array([[p_main[0]], [p_main[1]]], dtype=np.float64)
    pts_secondary = np.array([[p_secondary[0]], [p_secondary[1]]], dtype=np.float64)
    points_4d = cv2.triangulatePoints(model.p_main, model.p_secondary, pts_main, pts_secondary)
    w = points_4d[3, 0]
    if abs(w) < 1e-9:
        return None, float("inf")
    point = points_4d[:3, 0] / w

    ray_main = model.main_position
    dir_main = point - ray_main
    dir_main_norm = np.linalg.norm(dir_main)
    if dir_main_norm < 1e-9:
        return None, float("inf")
    dir_main = dir_main / dir_main_norm

    ray_secondary = model.secondary_position
    dir_secondary = point - ray_secondary
    dir_secondary_norm = np.linalg.norm(dir_secondary)
    if dir_secondary_norm < 1e-9:
        return None, float("inf")
    dir_secondary = dir_secondary / dir_secondary_norm

    w0 = ray_main - ray_secondary
    a = np.dot(dir_main, dir_main)
    b = np.dot(dir_main, dir_secondary)
    c = np.dot(dir_secondary, dir_secondary)
    d = np.dot(dir_main, w0)
    e = np.dot(dir_secondary, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        residual = float(np.linalg.norm(point - ray_main))
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
        closest_main = ray_main + sc * dir_main
        closest_secondary = ray_secondary + tc * dir_secondary
        residual = float(np.linalg.norm(closest_main - closest_secondary) / 2.0)

    if point[2] < 0.0 or point[2] > MAX_TRIANGULATION_HEIGHT_M:
        return None, residual
    if residual > MAX_TRIANGULATION_RESIDUAL_M:
        return None, residual
    return point, residual


def _polyline_length_3d(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def interpolate_track_at_frame(
    track: list[Point2D],
    frame: float,
) -> tuple[float, float] | None:
    """Linear interpolation of a 2D track at a (possibly fractional) frame index."""
    if not track:
        return None

    ordered = sorted(track, key=lambda point: point.frame)
    if frame < ordered[0].frame or frame > ordered[-1].frame:
        return None

    for index in range(len(ordered) - 1):
        start = ordered[index]
        end = ordered[index + 1]
        if start.frame <= frame <= end.frame:
            if end.frame == start.frame:
                return float(start.x), float(start.y)
            t = (frame - start.frame) / (end.frame - start.frame)
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            return x, y

    last = ordered[-1]
    if frame == last.frame:
        return float(last.x), float(last.y)
    return None


def _fit_curve_3d(
    points: list[Point3D],
    *,
    sample_count: int = 120,
) -> list[CurvePoint3D]:
    if len(points) < 3:
        return [CurvePoint3D(x=p.x, y=p.y, z=p.z) for p in points]

    frames = np.array([p.frame if p.frame is not None else i for i, p in enumerate(points)])
    xs = np.array([p.x for p in points], dtype=np.float64)
    ys = np.array([p.y for p in points], dtype=np.float64)
    zs = np.array([p.z for p in points], dtype=np.float64)

    try:
        cx = np.polyfit(frames, xs, 2)
        cy = np.polyfit(frames, ys, 2)
        cz = np.polyfit(frames, zs, 2)
    except np.linalg.LinAlgError:
        return [CurvePoint3D(x=p.x, y=p.y, z=p.z) for p in points]

    t_start = float(frames.min())
    t_end = float(frames.max())
    t_sample = np.linspace(t_start, t_end, sample_count)
    return [
        CurvePoint3D(
            x=float(np.polyval(cx, t)),
            y=float(np.polyval(cy, t)),
            z=float(np.polyval(cz, t)),
        )
        for t in t_sample
    ]


def triangulate_throw(
    left_track: list[Point2D],
    right_track: list[Point2D],
    *,
    setup: CameraSetup,
    frame_size: tuple[int, int],
    fps: float,
    throw_id: int,
    frame_offset: float | None = None,
) -> ThrowRecord | None:
    if not left_track or not right_track:
        return None

    width, height = frame_size
    model = build_stereo_camera_model(setup, width=width, height=height)

    right_by_frame = {p.frame: p for p in right_track}
    points_3d: list[Point3D] = []

    for left_point in left_track:
        if left_point.frame is None:
            continue
        if frame_offset is not None:
            secondary_coords = interpolate_track_at_frame(
                right_track,
                left_point.frame + frame_offset,
            )
            if secondary_coords is None:
                continue
            right_xy = secondary_coords
        else:
            right_point = right_by_frame.get(left_point.frame)
            if right_point is None:
                continue
            right_xy = (float(right_point.x), float(right_point.y))

        triangulated, _residual = _triangulate_point(
            np.array([left_point.x, left_point.y], dtype=np.float64),
            np.array(right_xy, dtype=np.float64),
            model,
        )
        if triangulated is None:
            continue
        points_3d.append(
            Point3D(
                frame=left_point.frame,
                x=float(triangulated[0]),
                y=float(triangulated[1]),
                z=float(triangulated[2]),
            )
        )

    if len(points_3d) < 3:
        return None

    start_frame = min(p.frame for p in left_track if p.frame is not None)
    end_frame = max(p.frame for p in left_track if p.frame is not None)
    fitted = _fit_curve_3d(points_3d)

    duration_s = (end_frame - start_frame + 1) / fps if fps > 0 else 0.0
    curve_len = _polyline_length_3d([(p.x, p.y, p.z) for p in fitted])
    speed_m_s = curve_len / duration_s if duration_s > 0 else None

    return ThrowRecord(
        id=throw_id,
        start_frame=start_frame,
        end_frame=end_frame,
        points_3d=points_3d,
        fitted_curve_3d=fitted,
        speed_m_s=speed_m_s,
        tracks_2d={"left": list(left_track), "right": list(right_track)},
    )
