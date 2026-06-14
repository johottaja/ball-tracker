from __future__ import annotations

import math
from collections import deque

from .config import ASSUMED_TORSO_CM, TORSO_LENGTH_BUFFER_SIZE


class TorsoLengthBuffer:
    """Rolling mean of shoulder-to-hip length in pixels."""

    def __init__(self, size: int = TORSO_LENGTH_BUFFER_SIZE) -> None:
        self._values: deque[float] = deque(maxlen=size)

    def reset(self) -> None:
        self._values.clear()

    def add(self, torso_length_px: float | None) -> None:
        if torso_length_px is not None and torso_length_px > 0:
            self._values.append(torso_length_px)

    @property
    def smoothed(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)


def polyline_length_px(points: list[tuple[int, int]]) -> float:
    """Sum of Euclidean segment lengths along a polyline."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def estimate_throw_speed_m_s(
    curve_points: list[tuple[int, int]] | None,
    torso_length_px: float | None,
    tracking_frames: int | None,
    video_fps: float | None,
) -> float | None:
    """Infer throw speed from fitted curve length and torso-based scale."""
    if curve_points is None or len(curve_points) < 2:
        return None
    if torso_length_px is None or torso_length_px <= 0:
        return None
    if tracking_frames is None or tracking_frames <= 0:
        return None
    if video_fps is None or video_fps <= 0:
        return None

    curve_px = polyline_length_px(curve_points)
    meters_per_px = (ASSUMED_TORSO_CM / 100.0) / torso_length_px
    distance_m = curve_px * meters_per_px
    duration_s = tracking_frames / video_fps
    if duration_s <= 0:
        return None
    return distance_m / duration_s
