from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from calibration import TableCalibration

from .config import MAX_TRIANGULATION_HEIGHT_M, MAX_TRIANGULATION_RESIDUAL_M
from .game_data import CurvePoint3D, Point2D, Point3D, ThrowRecord


@dataclass(frozen=True)
class StereoProjectionModel:
    """Cached 3×4 projection matrices for both calibrated cameras."""

    width: int
    height: int
    p_left: np.ndarray
    p_right: np.ndarray


def build_stereo_projection_model(
    calibration: TableCalibration,
    *,
    width: int,
    height: int,
) -> StereoProjectionModel:
    if calibration.image_width != width or calibration.image_height != height:
        raise ValueError(
            "Video frame size does not match calibration "
            f"({calibration.image_width}×{calibration.image_height} vs {width}×{height})"
        )
    left = calibration.camera("left")
    right = calibration.camera("right")
    if left is None or right is None:
        raise ValueError("Calibration must include left and right cameras")
    return StereoProjectionModel(
        width=width,
        height=height,
        p_left=left.projection_matrix,
        p_right=right.projection_matrix,
    )


def _triangulate_point(
    p_left: np.ndarray,
    p_right: np.ndarray,
    model: StereoProjectionModel,
) -> tuple[np.ndarray | None, float]:
    pts_left = np.array([[p_left[0]], [p_left[1]]], dtype=np.float64)
    pts_right = np.array([[p_right[0]], [p_right[1]]], dtype=np.float64)
    points_4d = cv2.triangulatePoints(model.p_left, model.p_right, pts_left, pts_right)
    w = points_4d[3, 0]
    if abs(w) < 1e-9:
        return None, float("inf")
    point = points_4d[:3, 0] / w

    # Epipolar residual via closest approach between the two viewing rays.
    c_left = _camera_center(model.p_left)
    c_right = _camera_center(model.p_right)
    if c_left is None or c_right is None:
        return None, float("inf")

    dir_left = point - c_left
    dir_right = point - c_right
    dir_left_norm = np.linalg.norm(dir_left)
    dir_right_norm = np.linalg.norm(dir_right)
    if dir_left_norm < 1e-9 or dir_right_norm < 1e-9:
        return None, float("inf")
    dir_left = dir_left / dir_left_norm
    dir_right = dir_right / dir_right_norm

    w0 = c_left - c_right
    a = np.dot(dir_left, dir_left)
    b = np.dot(dir_left, dir_right)
    c = np.dot(dir_right, dir_right)
    d = np.dot(dir_left, w0)
    e = np.dot(dir_right, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        residual = float(np.linalg.norm(point - c_left))
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
        closest_left = c_left + sc * dir_left
        closest_right = c_right + tc * dir_right
        residual = float(np.linalg.norm(closest_left - closest_right) / 2.0)

    if point[2] < 0.0 or point[2] > MAX_TRIANGULATION_HEIGHT_M:
        return None, residual
    if residual > MAX_TRIANGULATION_RESIDUAL_M:
        return None, residual
    return point, residual


def _camera_center(projection: np.ndarray) -> np.ndarray | None:
    """Camera center in world coordinates from a 3×4 projection matrix."""
    if projection.shape != (3, 4):
        return None
    r = projection[:, :3]
    t = projection[:, 3]
    det = np.linalg.det(r)
    if abs(det) < 1e-9:
        return None
    return -np.linalg.inv(r) @ t


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
    """Linear interpolation of a 2D track at a fractional frame index."""
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
    calibration: TableCalibration | None,
    frame_size: tuple[int, int],
    fps: float,
    throw_id: int,
    frame_offset: float | None = None,
) -> ThrowRecord | None:
    if calibration is None or not left_track or not right_track:
        return None

    width, height = frame_size
    try:
        model = build_stereo_projection_model(calibration, width=width, height=height)
    except ValueError:
        return None

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
