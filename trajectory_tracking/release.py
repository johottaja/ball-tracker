from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import numpy as np

from pose_detection import DominantHandDetection
from pose_detection.config import POSE_KEYPOINT_MIN_CONF

from .config import (
    PALM_EXTENSION,
    RELEASE_HIT_RADIUS_FACTOR,
    RELEASE_MAX_LOOKBACK_FRAMES,
)

Axis = Literal["x", "y"]


class FramePointLike(Protocol):
    frame: int
    x: int
    y: int


@dataclass(frozen=True)
class ParabolaFit:
    """Quadratic fit y = f(x) or x = f(y) for image-space trajectories."""

    axis: Axis
    coeffs: np.ndarray


@dataclass(frozen=True)
class ReleasePoint:
    frame: int
    x: int
    y: int


def palm_position(
    detection: DominantHandDetection | None,
    *,
    extension: float = PALM_EXTENSION,
) -> tuple[float, float] | None:
    """Estimate palm center as elbow + extension * (wrist - elbow)."""
    if detection is None:
        return None
    joints = detection.hand.joints
    if len(joints) < 3:
        return None
    elbow = joints[1]
    wrist = joints[2]
    if elbow.confidence < POSE_KEYPOINT_MIN_CONF or wrist.confidence < POSE_KEYPOINT_MIN_CONF:
        return None
    return (
        elbow.x + extension * (wrist.x - elbow.x),
        elbow.y + extension * (wrist.y - elbow.y),
    )


def forearm_length_px(detection: DominantHandDetection | None) -> float | None:
    if detection is None:
        return None
    joints = detection.hand.joints
    if len(joints) < 3:
        return None
    elbow = joints[1]
    wrist = joints[2]
    if elbow.confidence < POSE_KEYPOINT_MIN_CONF or wrist.confidence < POSE_KEYPOINT_MIN_CONF:
        return None
    return math.hypot(wrist.x - elbow.x, wrist.y - elbow.y)


def fit_parabola(points: Sequence[tuple[int, int]]) -> ParabolaFit | None:
    """Fit a degree-2 parabola to pixel points (same axis choice as TrajectoryTracker)."""
    if len(points) < 3:
        return None

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()

    try:
        if x_range >= y_range:
            return ParabolaFit(axis="x", coeffs=np.polyfit(xs, ys, 2))
        return ParabolaFit(axis="y", coeffs=np.polyfit(ys, xs, 2))
    except (np.linalg.LinAlgError, ValueError):
        return None


def sample_parabola(
    fit: ParabolaFit,
    points: Sequence[tuple[int, int]],
    *,
    sample_count: int = 120,
    extend_ratio: float = 0.15,
) -> list[tuple[int, int]]:
    """Sample a fitted parabola across the point span with optional end padding."""
    if len(points) < 2:
        return list(points)

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)

    if fit.axis == "x":
        x_range = xs.max() - xs.min()
        x_start = xs.min() - x_range * extend_ratio
        x_end = xs.max() + x_range * extend_ratio
        x_sample = np.linspace(x_start, x_end, sample_count)
        y_sample = np.polyval(fit.coeffs, x_sample)
        return [(int(x), int(y)) for x, y in zip(x_sample, y_sample)]

    y_range = ys.max() - ys.min()
    y_start = ys.min() - y_range * extend_ratio
    y_end = ys.max() + y_range * extend_ratio
    y_sample = np.linspace(y_start, y_end, sample_count)
    x_sample = np.polyval(fit.coeffs, y_sample)
    return [(int(x), int(y)) for x, y in zip(x_sample, y_sample)]


def _estimate_speed_px_per_frame(track: Sequence[FramePointLike]) -> float:
    if len(track) < 2:
        return 5.0
    speeds: list[float] = []
    for index in range(1, min(3, len(track))):
        previous = track[index - 1]
        current = track[index]
        delta_frames = current.frame - previous.frame
        if delta_frames <= 0:
            continue
        distance = math.hypot(current.x - previous.x, current.y - previous.y)
        speeds.append(distance / delta_frames)
    if not speeds:
        return 5.0
    return sum(speeds) / len(speeds)


def _backward_axis_sign(track: Sequence[FramePointLike], fit: ParabolaFit) -> float:
    """Step direction for walking backward along the independent variable."""
    if len(track) < 2:
        return -1.0
    first = track[0]
    second = track[1]
    if fit.axis == "x":
        return -1.0 if second.x >= first.x else 1.0
    return -1.0 if second.y >= first.y else 1.0


def position_on_parabola_backward(
    fit: ParabolaFit,
    anchor: tuple[float, float],
    arc_length: float,
    *,
    axis_sign: float,
    step: float = 0.5,
) -> tuple[float, float]:
    """Walk backward from anchor along the fitted parabola by arc_length pixels."""
    if arc_length <= 0:
        return anchor

    x, y = anchor
    traveled = 0.0
    while traveled < arc_length:
        if fit.axis == "x":
            x_new = x + axis_sign * step
            y_new = float(np.polyval(fit.coeffs, x_new))
        else:
            y_new = y + axis_sign * step
            x_new = float(np.polyval(fit.coeffs, y_new))
        segment = math.hypot(x_new - x, y_new - y)
        if segment < 1e-9:
            break
        traveled += segment
        x, y = x_new, y_new
    return x, y


def _throw_search_start_frame(
    first_frame: int,
    *,
    throw_label_at: Callable[[int], int],
    max_lookback: int,
) -> int:
    search_end = max(0, first_frame - max_lookback)
    for frame in range(first_frame, search_end - 1, -1):
        if throw_label_at(frame) != 1:
            return min(first_frame, frame + 1)
    return search_end


def find_release_point(
    track: Sequence[FramePointLike],
    fit: ParabolaFit | None,
    *,
    pose_at: Callable[[int], DominantHandDetection | None],
    throw_label_at: Callable[[int], int],
    max_lookback: int = RELEASE_MAX_LOOKBACK_FRAMES,
    palm_extension: float = PALM_EXTENSION,
    hit_radius_factor: float = RELEASE_HIT_RADIUS_FACTOR,
) -> ReleasePoint | None:
    """
    Backtrack along the fitted trajectory to find where it meets the thrower's palm.

    Uses cached per-frame pose (no YOLO reruns) and GRU throw labels to bound the search.
    """
    if not track or fit is None:
        return None

    first = track[0]
    anchor = (float(first.x), float(first.y))
    speed = _estimate_speed_px_per_frame(track)
    axis_sign = _backward_axis_sign(track, fit)
    search_end = _throw_search_start_frame(
        first.frame,
        throw_label_at=throw_label_at,
        max_lookback=max_lookback,
    )

    best_frame: int | None = None
    best_error = float("inf")
    best_position: tuple[float, float] | None = None
    best_forearm: float | None = None

    for frame in range(first.frame - 1, search_end - 1, -1):
        detection = pose_at(frame)
        palm = palm_position(detection, extension=palm_extension)
        if palm is None:
            continue

        arc_length = (first.frame - frame) * speed
        ball_position = position_on_parabola_backward(
            fit,
            anchor,
            arc_length,
            axis_sign=axis_sign,
        )
        error = math.hypot(ball_position[0] - palm[0], ball_position[1] - palm[1])
        if error < best_error:
            best_error = error
            best_frame = frame
            best_position = ball_position
            best_forearm = forearm_length_px(detection)

    if best_frame is None or best_position is None:
        return None

    radius_limit = (
        hit_radius_factor * best_forearm
        if best_forearm is not None and best_forearm > 0
        else 60.0
    )
    if best_error > radius_limit:
        return None

    return ReleasePoint(
        frame=best_frame,
        x=int(round(best_position[0])),
        y=int(round(best_position[1])),
    )


def find_secondary_release_at_frame(
    track: Sequence[FramePointLike],
    fit: ParabolaFit | None,
    release_frame: int,
    *,
    timeline_offset: float = 0.0,
) -> ReleasePoint | None:
    """Extrapolate the secondary-camera parabola to a main-timeline release frame."""
    if not track or fit is None:
        return None

    secondary_release_frame = release_frame + timeline_offset
    if secondary_release_frame >= track[0].frame:
        return None

    first = track[0]
    anchor = (float(first.x), float(first.y))
    speed = _estimate_speed_px_per_frame(track)
    axis_sign = _backward_axis_sign(track, fit)
    arc_length = (first.frame - secondary_release_frame) * speed
    position = position_on_parabola_backward(
        fit,
        anchor,
        arc_length,
        axis_sign=axis_sign,
    )
    return ReleasePoint(
        frame=int(round(secondary_release_frame)),
        x=int(round(position[0])),
        y=int(round(position[1])),
    )


def _pose_at_from_cache(cache: object, frame: int) -> DominantHandDetection | None:
    if not cache.has_pose(frame):
        return None
    return cache.get_pose(frame)


def _throw_label_at_from_cache(cache: object, frame: int) -> int:
    if not cache.has_gru(frame):
        return 0
    return cache.get_gru(frame).label


def find_release_point_from_cache(
    track: Sequence[FramePointLike],
    fit: ParabolaFit | None,
    cache: object,
) -> ReleasePoint | None:
    return find_release_point(
        track,
        fit,
        pose_at=lambda frame: _pose_at_from_cache(cache, frame),
        throw_label_at=lambda frame: _throw_label_at_from_cache(cache, frame),
    )


def prepend_release_to_track(
    track: list[FramePointLike],
    release: ReleasePoint,
) -> list[FramePointLike]:
    """Return a new track with the release point inserted at the front."""
    release_point = type(track[0])(frame=release.frame, x=release.x, y=release.y)
    return [release_point, *track]


def apply_secondary_release_extension(
    trajectory: list[tuple[int, int]],
    trajectory_frames: list[int] | None,
    fit: ParabolaFit | None,
    release_frame: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]] | None]:
    """Prepend secondary-camera release point aligned to the main-camera release frame."""
    if not trajectory or trajectory_frames is None or len(trajectory_frames) != len(trajectory):
        return trajectory, None
    indexed = [
        _IndexedPoint(frame=frame, x=point[0], y=point[1])
        for frame, point in zip(trajectory_frames, trajectory)
    ]
    release = find_secondary_release_at_frame(indexed, fit, release_frame)
    if release is None:
        return trajectory, None

    extended = [(release.x, release.y), *trajectory]
    extended_fit = fit_parabola(extended) or fit
    curve = sample_parabola(extended_fit, extended) if extended_fit is not None else None
    return extended, curve


def extend_completed_trajectory(
    trajectory: list[tuple[int, int]],
    trajectory_frames: list[int] | None,
    fit: ParabolaFit | None,
    cache: object | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]] | None, ReleasePoint | None]:
    """
    Prepend a release point to a completed trajectory and resample the display curve.

    Returns (extended_points, extended_curve_samples, release_point).
    """
    if not trajectory or trajectory_frames is None or len(trajectory_frames) != len(trajectory):
        return trajectory, None, None
    if cache is None or fit is None:
        return trajectory, None, None

    indexed = [
        _IndexedPoint(frame=frame, x=point[0], y=point[1])
        for frame, point in zip(trajectory_frames, trajectory)
    ]
    release = find_release_point_from_cache(indexed, fit, cache)
    if release is None:
        return trajectory, None, None

    extended = [(release.x, release.y), *trajectory]
    extended_fit = fit_parabola(extended) or fit
    curve = sample_parabola(extended_fit, extended)
    return extended, curve, release


@dataclass(frozen=True)
class _IndexedPoint:
    frame: int
    x: int
    y: int
