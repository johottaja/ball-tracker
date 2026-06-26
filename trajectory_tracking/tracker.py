from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config import (
    BALL_CIRCULARITY_MAX,
    BALL_CIRCULARITY_MIN,
    BALL_CONTOUR_MIN_AREA,
    MIN_TRAJECTORY_POINTS,
    SECTOR_ANGLE_DEG,
    SECTOR_DIRECTION_DEG,
    SECTOR_RADIUS_PX,
    TRACKING_TIMEOUT_FRAMES,
)


class Phase(str, Enum):
    DETECTING_THROW = "detecting_throw"
    SCANNING_BALL = "scanning_ball"
    TRACKING_BALL = "tracking_ball"
    AWAITING_PARTNER = "awaiting_partner"


@dataclass(frozen=True)
class TrajectoryResult:
    """Snapshot of tracker state returned after each update call."""

    phase: Phase

    # Sector to visualise this frame (present in SCANNING_BALL and TRACKING_BALL).
    scan_origin: tuple[int, int] | None
    # Direction the sector points (degrees, screen coords: 0=right, 90=down).
    scan_direction_deg: float | None

    # Ball position found in the current frame (None if not detected this frame).
    detected_ball_pos: tuple[int, int] | None

    # Points accumulated during the current TRACKING_BALL phase.
    trajectory_points: list[tuple[int, int]]

    # Points from the most recently completed trajectory.
    completed_trajectory: list[tuple[int, int]] | None

    # Sampled parabola points for the completed trajectory (None if < 3 points).
    fitted_curve_points: list[tuple[int, int]] | None

    # Frames spent in TRACKING_BALL for the most recently completed trajectory.
    completed_tracking_frames: int | None

    # Increments each time a trajectory is finalised (used to detect new completions).
    completion_id: int


class TrajectoryTracker:
    """
    Three-phase stateful ball trajectory tracker.

    Phase 1 – DETECTING_THROW: waits for throw_label == 1.
    Phase 2 – SCANNING_BALL:   searches a circular sector at the wrist for
                                a circular contour; resets on every new label-1.
    Phase 3 – TRACKING_BALL:   records ball positions; ends after
                                `timeout_frames` consecutive misses and fits a
                                parabola to the collected points.
    """

    def __init__(
        self,
        *,
        sector_angle_deg: float = SECTOR_ANGLE_DEG,
        sector_direction_deg: float = SECTOR_DIRECTION_DEG,
        sector_radius: int = SECTOR_RADIUS_PX,
        timeout_frames: int = TRACKING_TIMEOUT_FRAMES,
    ) -> None:
        self._sector_half_angle = sector_angle_deg / 2.0
        self._sector_direction_deg = sector_direction_deg
        self.sector_radius = sector_radius
        self.timeout_frames = timeout_frames

        self.phase = Phase.DETECTING_THROW
        self._scan_origin: tuple[int, int] | None = None
        self._miss_count: int = 0
        self._trajectory_points: list[tuple[int, int]] = []
        self._completed_trajectory: list[tuple[int, int]] | None = None
        self._fitted_curve_points: list[tuple[int, int]] | None = None
        self._tracking_frame_count: int = 0
        self._completed_tracking_frames: int | None = None
        self._completion_id: int = 0
        self._full_frame_scan: bool = False

    def reset(self) -> None:
        self.phase = Phase.DETECTING_THROW
        self._scan_origin = None
        self._full_frame_scan = False
        self._miss_count = 0
        self._trajectory_points = []
        self._completed_trajectory = None
        self._fitted_curve_points = None
        self._tracking_frame_count = 0
        self._completed_tracking_frames = None
        self._completion_id = 0

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        throw_label: int,
        wrist_pos: tuple[int, int] | None,
        motion_mask: np.ndarray | None,
        *,
        defer_detecting_throw: bool = False,
    ) -> TrajectoryResult:
        """Advance the tracker by one frame and return a result snapshot."""
        if self.phase == Phase.AWAITING_PARTNER:
            return self._result_snapshot(None)

        detected_pos: tuple[int, int] | None = None

        if self.phase == Phase.DETECTING_THROW:
            if throw_label == 1 and wrist_pos is not None:
                self._enter_scanning(wrist_pos)

        elif self.phase == Phase.SCANNING_BALL:
            if throw_label == 1 and wrist_pos is not None:
                # Keep resetting the wrist anchor while the throw is still on.
                self._enter_scanning(wrist_pos)

            if motion_mask is not None and self._scan_origin is not None:
                detected_pos = self._find_ball(motion_mask)
                if detected_pos is not None:
                    # First detection – transition to tracking.
                    self._scan_origin = detected_pos
                    self._trajectory_points = [detected_pos]
                    self._miss_count = 0
                    self._tracking_frame_count = 1
                    self.phase = Phase.TRACKING_BALL

        else:  # TRACKING_BALL
            self._tracking_frame_count += 1
            if throw_label == 1 and wrist_pos is not None:
                # A new throw starts while we were tracking – finalise and restart.
                self._finalize_trajectory()
                self._enter_scanning(wrist_pos)
            else:
                if motion_mask is not None:
                    detected_pos = self._find_ball(motion_mask)

                if detected_pos is not None:
                    self._scan_origin = detected_pos
                    self._trajectory_points.append(detected_pos)
                    self._miss_count = 0
                else:
                    self._miss_count += 1
                    if self._miss_count >= self.timeout_frames:
                        self._finalize_trajectory()
                        self.phase = (
                            Phase.AWAITING_PARTNER
                            if defer_detecting_throw
                            else Phase.DETECTING_THROW
                        )

        return self._result_snapshot(detected_pos)

    def update_secondary(
        self,
        throw_label: int,
        motion_mask: np.ndarray | None,
        *,
        defer_detecting_throw: bool = False,
    ) -> TrajectoryResult:
        """
        Ball-only tracker for a secondary camera view.

        Throw detection comes from the main camera (``throw_label`` only); ball
        search starts with a full-frame scan because there is no wrist anchor.
        """
        if self.phase == Phase.AWAITING_PARTNER:
            return self._result_snapshot(None)

        detected_pos: tuple[int, int] | None = None

        if self.phase == Phase.DETECTING_THROW:
            if throw_label == 1:
                self._enter_scanning_full_frame()

        elif self.phase == Phase.SCANNING_BALL:
            if motion_mask is not None:
                detected_pos = self._find_ball(motion_mask)
                if detected_pos is not None:
                    self._full_frame_scan = False
                    self._scan_origin = detected_pos
                    self._trajectory_points = [detected_pos]
                    self._miss_count = 0
                    self._tracking_frame_count = 1
                    self.phase = Phase.TRACKING_BALL

        else:  # TRACKING_BALL
            self._tracking_frame_count += 1
            if throw_label == 1:
                self._finalize_trajectory()
                self._enter_scanning_full_frame()
            else:
                if motion_mask is not None:
                    detected_pos = self._find_ball(motion_mask)

                if detected_pos is not None:
                    self._scan_origin = detected_pos
                    self._trajectory_points.append(detected_pos)
                    self._miss_count = 0
                else:
                    self._miss_count += 1
                    if self._miss_count >= self.timeout_frames:
                        self._finalize_trajectory()
                        self.phase = (
                            Phase.AWAITING_PARTNER
                            if defer_detecting_throw
                            else Phase.DETECTING_THROW
                        )

        return self._result_snapshot(detected_pos)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_snapshot(
        self, detected_pos: tuple[int, int] | None
    ) -> TrajectoryResult:
        return TrajectoryResult(
            phase=self.phase,
            scan_origin=self._scan_origin,
            scan_direction_deg=self._sector_direction_deg,
            detected_ball_pos=detected_pos,
            trajectory_points=list(self._trajectory_points),
            completed_trajectory=self._completed_trajectory,
            fitted_curve_points=self._fitted_curve_points,
            completed_tracking_frames=self._completed_tracking_frames,
            completion_id=self._completion_id,
        )

    def _enter_scanning(self, wrist_pos: tuple[int, int]) -> None:
        self.phase = Phase.SCANNING_BALL
        self._scan_origin = wrist_pos
        self._full_frame_scan = False
        self._trajectory_points = []
        self._miss_count = 0

    def _enter_scanning_full_frame(self) -> None:
        self.phase = Phase.SCANNING_BALL
        self._scan_origin = None
        self._full_frame_scan = True
        self._trajectory_points = []
        self._miss_count = 0

    def _finalize_trajectory(self) -> None:
        """Fit a parabola to collected points and save the completed trajectory."""
        points = self._trajectory_points
        tracking_frames = self._tracking_frame_count
        self._trajectory_points = []
        self._tracking_frame_count = 0

        if len(points) < MIN_TRAJECTORY_POINTS:
            return

        self._completed_trajectory = list(points)
        self._fitted_curve_points = None

        if len(points) >= 3:
            xs = np.array([p[0] for p in points], dtype=np.float64)
            ys = np.array([p[1] for p in points], dtype=np.float64)
            x_range = xs.max() - xs.min()
            y_range = ys.max() - ys.min()

            try:
                if x_range >= y_range:
                    # y = f(x)
                    coeffs = np.polyfit(xs, ys, 2)
                    x_start = xs.min() - x_range * 0.15
                    x_end = xs.max() + x_range * 0.15
                    x_sample = np.linspace(x_start, x_end, 120)
                    y_sample = np.polyval(coeffs, x_sample)
                    self._fitted_curve_points = [
                        (int(x), int(y)) for x, y in zip(x_sample, y_sample)
                    ]
                else:
                    # x = f(y) – for near-vertical throws
                    coeffs = np.polyfit(ys, xs, 2)
                    y_start = ys.min() - y_range * 0.15
                    y_end = ys.max() + y_range * 0.15
                    y_sample = np.linspace(y_start, y_end, 120)
                    x_sample = np.polyval(coeffs, y_sample)
                    self._fitted_curve_points = [
                        (int(x), int(y)) for x, y in zip(x_sample, y_sample)
                    ]
            except (np.linalg.LinAlgError, ValueError):
                pass

        self._completed_tracking_frames = tracking_frames
        self._completion_id += 1

    def _find_ball(self, motion_mask: np.ndarray) -> tuple[int, int] | None:
        """Return the centroid of the largest circular contour in the search area."""
        if not self._full_frame_scan and self._scan_origin is None:
            return None

        gray = (
            cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)
            if motion_mask.ndim == 3
            else motion_mask
        )
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: tuple[int, int] | None = None
        best_area = 0.0

        for contour in contours:
            if not self._is_circular(contour):
                continue
            cx, cy = self._centroid(contour)
            if not self._full_frame_scan and not self._in_sector(cx, cy):
                continue
            area = cv2.contourArea(contour)
            if area > best_area:
                best_area = area
                best = (cx, cy)

        return best

    def _is_circular(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        if area < BALL_CONTOUR_MIN_AREA:
            return False
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = (4 * math.pi * area) / (perimeter**2)
        return BALL_CIRCULARITY_MIN < circularity <= BALL_CIRCULARITY_MAX

    @staticmethod
    def _centroid(contour: np.ndarray) -> tuple[int, int]:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            x, y, w, h = cv2.boundingRect(contour)
            return x + w // 2, y + h // 2
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    def _in_sector(self, x: int, y: int) -> bool:
        ox, oy = self._scan_origin
        dx, dy = x - ox, y - oy
        dist = math.sqrt(dx**2 + dy**2)
        if dist > self.sector_radius:
            return False
        point_angle = math.degrees(math.atan2(dy, dx))
        diff = (point_angle - self._sector_direction_deg + 180) % 360 - 180
        return abs(diff) <= self._sector_half_angle
