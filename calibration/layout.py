from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .types import (
    CalibrationLayout,
    CameraLayoutStats,
    StereoLayoutStats,
    TableCalibration,
)

_CAMERA_COLORS = {"left": "#cc2222", "right": "#2255cc"}
ScreenSide = Literal["left", "right"]


@dataclass(frozen=True)
class StereoScreenSideMapping:
    """How main-camera image halves correspond to secondary image halves."""

    main_left_to_secondary: ScreenSide
    horizontal_axis_dot: float


@dataclass(frozen=True)
class CameraLayoutInfo:
    name: str
    color: str
    center: tuple[float, float, float]
    xy_distance_m: float
    z_m: float
    yaw_deg: float
    pitch_deg: float
    horizontal_fov_deg: float
    fov_left_xy: tuple[float, float] | None
    fov_right_xy: tuple[float, float] | None


def camera_center_from_projection(projection: np.ndarray) -> np.ndarray | None:
    """World-space camera center from a 3×4 projection matrix P = K[R|t]."""
    if projection.shape != (3, 4):
        return None
    rotation = projection[:, :3]
    translation = projection[:, 3]
    det = np.linalg.det(rotation)
    if abs(det) < 1e-9:
        return None
    return -np.linalg.inv(rotation) @ translation


def _image_ray_direction(projection: np.ndarray, u: float, v: float) -> np.ndarray | None:
    center = camera_center_from_projection(projection)
    if center is None:
        return None
    rotation = projection[:, :3]
    det = np.linalg.det(rotation)
    if abs(det) < 1e-9:
        return None
    # For P = M[X | 1], the back-projected ray is
    # X(s) = C + s M^-1 [u, v, 1]^T, where C = -M^-1 p4.
    # M^-1 [u, v, 1]^T is therefore already the world-space direction;
    # subtracting C treats it as a point and corrupts FOV and pose readouts.
    direction = np.linalg.inv(rotation) @ np.array([u, v, 1.0], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    return direction / norm


def _image_right_world_axis(projection: np.ndarray) -> np.ndarray | None:
    """Camera image-right unit vector expressed in world coordinates."""
    matrix = projection[:, :3]
    if matrix.shape != (3, 3) or abs(np.linalg.det(matrix)) < 1e-9:
        return None
    axis = np.linalg.inv(matrix) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = np.linalg.norm(axis)
    return axis / norm if norm >= 1e-9 else None


def infer_stereo_screen_side_mapping(
    calibration: TableCalibration,
    *,
    ambiguity_threshold: float = 0.25,
) -> StereoScreenSideMapping | None:
    """
    Infer whether image-left in the main camera is left or right in the secondary.

    P = K[R|t] maps the camera's local +X (image-right) direction into world
    space through the inverse 3×3 camera matrix. Aligned right axes preserve
    screen sides; opposed axes reverse them. Nearly perpendicular axes cannot
    reliably map two image halves and are rejected.
    """
    main = calibration.camera("left")
    secondary = calibration.camera("right")
    if main is None or secondary is None:
        return None
    main_axis = _image_right_world_axis(main.projection_matrix)
    secondary_axis = _image_right_world_axis(secondary.projection_matrix)
    if main_axis is None or secondary_axis is None:
        return None
    dot = float(np.dot(main_axis, secondary_axis))
    if abs(dot) < ambiguity_threshold:
        return None
    table_axis = np.array([main_axis[0], main_axis[1]], dtype=np.float64)
    table_norm = np.linalg.norm(table_axis)
    if table_norm < 1e-9:
        return None
    table_axis /= table_norm
    distance = min(calibration.table_length_m, calibration.table_width_m) * 0.25

    def projected_u(projection: np.ndarray, sign: float) -> float | None:
        world = np.array(
            [sign * distance * table_axis[0], sign * distance * table_axis[1], 0.0, 1.0],
            dtype=np.float64,
        )
        image = projection @ world
        if abs(image[2]) < 1e-9:
            return None
        return float(image[0] / image[2])

    main_delta = projected_u(main.projection_matrix, 1.0)
    main_origin = projected_u(main.projection_matrix, -1.0)
    secondary_delta = projected_u(secondary.projection_matrix, 1.0)
    secondary_origin = projected_u(secondary.projection_matrix, -1.0)
    if None in (main_delta, main_origin, secondary_delta, secondary_origin):
        return None
    projected_dot = (main_delta - main_origin) * (secondary_delta - secondary_origin)
    if abs(projected_dot) < 1e-9 or (projected_dot > 0) != (dot > 0):
        return None
    return StereoScreenSideMapping(
        main_left_to_secondary="left" if dot > 0 else "right",
        horizontal_axis_dot=dot,
    )


def _ray_table_xy_intersection(
    center: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float] | None:
    if abs(direction[2]) < 1e-9:
        return None
    distance = -center[2] / direction[2]
    if distance < 0:
        return None
    point = center + distance * direction
    return float(point[0]), float(point[1])


def _ray_xy_endpoint(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    length_m: float,
) -> tuple[float, float]:
    horizontal = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
    horizontal_norm = np.linalg.norm(horizontal)
    if horizontal_norm < 1e-9:
        horizontal = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        if np.linalg.norm(horizontal) < 1e-9:
            return float(center[0]), float(center[1])
    else:
        horizontal = horizontal / horizontal_norm
    end = center[:2] + horizontal[:2] * length_m
    return float(end[0]), float(end[1])


def _fov_edge_xy(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    fallback_length_m: float,
) -> tuple[float, float]:
    hit = _ray_table_xy_intersection(center, direction)
    if hit is not None:
        return hit
    return _ray_xy_endpoint(center, direction, length_m=fallback_length_m)


def _layout_info_to_stats(layout: CameraLayoutInfo) -> CameraLayoutStats:
    return CameraLayoutStats(
        name=layout.name,
        center=layout.center,
        xy_distance_m=layout.xy_distance_m,
        z_m=layout.z_m,
        yaw_deg=layout.yaw_deg,
        pitch_deg=layout.pitch_deg,
        horizontal_fov_deg=layout.horizontal_fov_deg,
        fov_left_xy=layout.fov_left_xy,
        fov_right_xy=layout.fov_right_xy,
    )


def _layout_stats_to_info(stats: CameraLayoutStats) -> CameraLayoutInfo:
    return CameraLayoutInfo(
        name=stats.name,
        color=_CAMERA_COLORS.get(stats.name, "#444444"),
        center=stats.center,
        xy_distance_m=stats.xy_distance_m,
        z_m=stats.z_m,
        yaw_deg=stats.yaw_deg,
        pitch_deg=stats.pitch_deg,
        horizontal_fov_deg=stats.horizontal_fov_deg,
        fov_left_xy=stats.fov_left_xy,
        fov_right_xy=stats.fov_right_xy,
    )


def compute_stereo_layout_stats(
    layouts: list[CameraLayoutInfo],
) -> StereoLayoutStats | None:
    left = next((item for item in layouts if item.name == "left"), None)
    right = next((item for item in layouts if item.name == "right"), None)
    if left is None or right is None:
        return None
    dx = right.center[0] - left.center[0]
    dy = right.center[1] - left.center[1]
    dz = right.center[2] - left.center[2]
    return StereoLayoutStats(
        baseline_xy_m=(dx * dx + dy * dy) ** 0.5,
        baseline_3d_m=(dx * dx + dy * dy + dz * dz) ** 0.5,
        delta_z_m=dz,
    )


def compute_calibration_layout(calibration: TableCalibration) -> CalibrationLayout:
    layouts = compute_camera_layout(calibration)
    return CalibrationLayout(
        cameras=[_layout_info_to_stats(layout) for layout in layouts],
        stereo=compute_stereo_layout_stats(layouts),
    )


def layout_info_from_calibration(calibration: TableCalibration) -> list[CameraLayoutInfo]:
    """Return saved layout stats when present, otherwise compute from projection matrices."""
    layout = calibration.layout
    if layout is not None and layout.cameras:
        return [_layout_stats_to_info(stats) for stats in layout.cameras]
    return compute_camera_layout(calibration)


def compute_camera_layout(calibration: TableCalibration) -> list[CameraLayoutInfo]:
    """Top-down layout data for each calibrated camera."""
    fallback_length_m = max(calibration.table_length_m, calibration.table_width_m) * 2.0
    image_width = calibration.image_width
    image_height = calibration.image_height
    center_y = image_height / 2.0

    layouts: list[CameraLayoutInfo] = []
    for camera in calibration.cameras:
        projection = camera.projection_matrix
        center = camera_center_from_projection(projection)
        if center is None:
            continue

        optical_axis = _image_ray_direction(projection, image_width / 2.0, center_y)
        if optical_axis is None:
            continue

        left_ray = _image_ray_direction(projection, 0.0, center_y)
        right_ray = _image_ray_direction(projection, float(image_width), center_y)
        if left_ray is None or right_ray is None:
            continue

        horizontal_fov_deg = math.degrees(
            math.acos(float(np.clip(np.dot(left_ray, right_ray), -1.0, 1.0)))
        )
        yaw_deg = math.degrees(math.atan2(float(optical_axis[1]), float(optical_axis[0])))
        horizontal_speed = math.hypot(float(optical_axis[0]), float(optical_axis[1]))
        pitch_deg = math.degrees(math.atan2(float(optical_axis[2]), horizontal_speed))

        xy_distance_m = math.hypot(float(center[0]), float(center[1]))
        fov_left_xy = _fov_edge_xy(
            center,
            left_ray,
            fallback_length_m=fallback_length_m,
        )
        fov_right_xy = _fov_edge_xy(
            center,
            right_ray,
            fallback_length_m=fallback_length_m,
        )

        layouts.append(
            CameraLayoutInfo(
                name=camera.name,
                color=_CAMERA_COLORS.get(camera.name, "#444444"),
                center=(float(center[0]), float(center[1]), float(center[2])),
                xy_distance_m=xy_distance_m,
                z_m=float(center[2]),
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                horizontal_fov_deg=horizontal_fov_deg,
                fov_left_xy=fov_left_xy,
                fov_right_xy=fov_right_xy,
            )
        )

    return layouts
