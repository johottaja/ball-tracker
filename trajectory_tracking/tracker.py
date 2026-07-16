from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config import (
    AWAITING_PARTNER_TIMEOUT_FRAMES,
    BALL_CIRCULARITY_MAX,
    BALL_CIRCULARITY_MIN,
    BALL_CONTOUR_MIN_AREA,
    BOUNCE_MISS_MIN_POINTS,
    MIN_TRAJECTORY_POINTS,
    SCANNING_TIMEOUT_FRAMES,
    SECTOR_ANGLE_DEG,
    SECTOR_DIRECTION_DEG,
    SECTOR_RADIUS_PX,
    TRACKING_TIMEOUT_FRAMES,
)
from .release import ParabolaFit, fit_parabola, sample_parabola


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

    # Quadratic fit for the completed trajectory (None if < 3 points).
    completed_parabola_fit: ParabolaFit | None

    # Frame index per completed trajectory point (parallel to completed_trajectory).
    completed_trajectory_frames: list[int] | None

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
        scanning_timeout_frames: int = SCANNING_TIMEOUT_FRAMES,
        awaiting_partner_timeout_frames: int = AWAITING_PARTNER_TIMEOUT_FRAMES,
    ) -> None:
        self._sector_half_angle = sector_angle_deg / 2.0
        self._sector_direction_deg = sector_direction_deg
        self.sector_radius = sector_radius
        self.timeout_frames = timeout_frames
        self.scanning_timeout_frames = scanning_timeout_frames
        self.awaiting_partner_timeout_frames = awaiting_partner_timeout_frames

        self.phase = Phase.DETECTING_THROW
        self._scan_origin: tuple[int, int] | None = None
        self._scan_frame_count: int = 0
        self._awaiting_frame_count: int = 0
        self._miss_count: int = 0
        self._trajectory_points: list[tuple[int, int]] = []
        self._trajectory_frames: list[int] = []
        self._completed_trajectory: list[tuple[int, int]] | None = None
        self._completed_trajectory_frames: list[int] | None = None
        self._fitted_curve_points: list[tuple[int, int]] | None = None
        self._completed_parabola_fit: ParabolaFit | None = None
        self._tracking_frame_count: int = 0
        self._completed_tracking_frames: int | None = None
        self._completion_id: int = 0
        self._full_frame_scan: bool = False
        self._stereo_reconcile: bool = False
        self._awaiting_valid_completion: bool = False

    def reset(self) -> None:
        self.phase = Phase.DETECTING_THROW
        self._scan_origin = None
        self._full_frame_scan = False
        self._miss_count = 0
        self._trajectory_points = []
        self._trajectory_frames = []
        self._completed_trajectory = None
        self._completed_trajectory_frames = None
        self._fitted_curve_points = None
        self._completed_parabola_fit = None
        self._tracking_frame_count = 0
        self._completed_tracking_frames = None
        self._completion_id = 0
        self._scan_frame_count = 0
        self._awaiting_frame_count = 0
        self._stereo_reconcile = False
        self._awaiting_valid_completion = False

    def set_sector_direction_deg(self, direction_deg: float) -> None:
        """Set the fixed sector direction for the currently selected player."""
        self._sector_direction_deg = direction_deg % 360.0

    def exit_awaiting_partner(self) -> None:
        """Leave AWAITING_PARTNER and return to throw detection."""
        self.phase = Phase.DETECTING_THROW
        self._awaiting_frame_count = 0
        self._awaiting_valid_completion = False

    def pop_stereo_reconcile(self) -> bool:
        """Return whether this camera needs to follow the partner phase this frame."""
        if not self._stereo_reconcile:
            return False
        self._stereo_reconcile = False
        return True

    def ball_search_origin(self) -> tuple[int, int] | None:
        """Best origin for (re)joining ball search: last track point or scan anchor."""
        if self._trajectory_points:
            return self._trajectory_points[-1]
        return self._scan_origin

    def adopt_partner_phase(
        self,
        partner: TrajectoryTracker,
        *,
        is_secondary: bool,
        throw_label: int,
        wrist_pos: tuple[int, int] | None,
    ) -> None:
        """Mirror the partner's phase after a failed local throw or scan."""
        partner_phase = partner.phase

        if partner_phase == Phase.AWAITING_PARTNER:
            self._enter_awaiting_partner(valid_completion=False)
            return

        if partner_phase == Phase.TRACKING_BALL:
            if is_secondary:
                if throw_label == 1 and wrist_pos is not None:
                    self._enter_scanning(wrist_pos)
                else:
                    self._enter_scanning_full_frame()
                return
            origin = partner.ball_search_origin()
            if origin is None:
                self.exit_awaiting_partner()
                return
            if throw_label == 1 and wrist_pos is not None:
                self._enter_scanning(wrist_pos)
            else:
                self._enter_scanning_at_origin(origin)
            return

        if partner_phase == Phase.SCANNING_BALL:
            if is_secondary:
                if throw_label == 1 and wrist_pos is not None:
                    self._enter_scanning(wrist_pos)
                else:
                    self._enter_scanning_full_frame()
                return
            if throw_label == 1 and wrist_pos is not None:
                self._enter_scanning(wrist_pos)
            elif partner._scan_origin is not None:
                self._enter_scanning_at_origin(partner._scan_origin)
            else:
                self.exit_awaiting_partner()
            return

        self.exit_awaiting_partner()

    def _mark_stereo_reconcile(self) -> None:
        self._stereo_reconcile = True
        self.phase = Phase.DETECTING_THROW
        self._scan_origin = None
        self._full_frame_scan = False

    def _complete_throw_stereo(self, saved: bool) -> None:
        if saved:
            self._enter_awaiting_partner()
        else:
            self._mark_stereo_reconcile()

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        throw_label: int,
        wrist_pos: tuple[int, int] | None,
        motion_mask: np.ndarray | None,
        *,
        alternate_motion_mask: np.ndarray | None = None,
        defer_detecting_throw: bool = False,
        frame_index: int | None = None,
    ) -> TrajectoryResult:
        """Advance the tracker by one frame and return a result snapshot."""
        if self.phase == Phase.AWAITING_PARTNER:
            self._awaiting_frame_count += 1
            if self._awaiting_frame_count >= self.awaiting_partner_timeout_frames:
                self.exit_awaiting_partner()
            return self._result_snapshot(None)

        detected_pos: tuple[int, int] | None = None

        if self.phase == Phase.DETECTING_THROW:
            if throw_label == 1 and wrist_pos is not None:
                self._enter_scanning(wrist_pos)

        elif self.phase == Phase.SCANNING_BALL:
            if throw_label == 1:
                self._scan_frame_count = 0
                if wrist_pos is not None:
                    self._scan_origin = wrist_pos

            if self._scan_origin is not None:
                detected_pos = self._find_ball_merged(motion_mask, alternate_motion_mask)
                if detected_pos is not None:
                    # First detection – transition to tracking.
                    self._scan_origin = detected_pos
                    self._trajectory_points = [detected_pos]
                    self._trajectory_frames = (
                        [frame_index] if frame_index is not None else []
                    )
                    self._miss_count = 0
                    self._tracking_frame_count = 1
                    self.phase = Phase.TRACKING_BALL

            if self.phase == Phase.SCANNING_BALL and throw_label == 0:
                self._scan_frame_count += 1
                if self._scan_frame_count >= self.scanning_timeout_frames:
                    self._exit_scanning_failed(defer_detecting_throw)

        else:  # TRACKING_BALL
            self._tracking_frame_count += 1
            if throw_label == 1 and wrist_pos is not None:
                saved = self._finalize_trajectory()
                if defer_detecting_throw:
                    if saved:
                        self._enter_awaiting_partner()
                    else:
                        self._mark_stereo_reconcile()
                else:
                    self._enter_scanning(wrist_pos)
            else:
                detected_pos = self._find_ball_merged(motion_mask, alternate_motion_mask)

                if detected_pos is not None:
                    if self._accept_tracking_detection(detected_pos, frame_index):
                        self._miss_count = 0
                    else:
                        self._miss_count += 1
                else:
                    self._miss_count += 1
                if self._miss_count >= self.timeout_frames:
                    saved = self._finalize_trajectory()
                    if defer_detecting_throw:
                        self._complete_throw_stereo(saved)
                    else:
                        self.phase = Phase.DETECTING_THROW

        return self._result_snapshot(detected_pos)

    def apply_release_extension(self, cache: object | None) -> None:
        """Prepend a release point to the last completed trajectory using cached pose."""
        if self._completed_trajectory is None:
            return
        from .release import extend_completed_trajectory

        extended, curve, release = extend_completed_trajectory(
            self._completed_trajectory,
            self._completed_trajectory_frames,
            self._completed_parabola_fit,
            cache,
        )
        if release is None:
            return
        self._completed_trajectory = extended
        if self._completed_trajectory_frames is not None:
            self._completed_trajectory_frames = [
                release.frame,
                *self._completed_trajectory_frames,
            ]
        if curve is not None:
            self._fitted_curve_points = curve
            refit = fit_parabola(extended)
            if refit is not None:
                self._completed_parabola_fit = refit

    def apply_secondary_release_extension(self, release_frame: int) -> None:
        """Prepend a release point extrapolated to match the main camera release frame."""
        if self._completed_trajectory is None:
            return
        from .release import apply_secondary_release_extension as extend_secondary

        extended, curve = extend_secondary(
            self._completed_trajectory,
            self._completed_trajectory_frames,
            self._completed_parabola_fit,
            release_frame,
        )
        if curve is None:
            return
        self._completed_trajectory = extended
        if self._completed_trajectory_frames is not None:
            self._completed_trajectory_frames = [release_frame, *self._completed_trajectory_frames]
        self._fitted_curve_points = curve
        refit = fit_parabola(extended)
        if refit is not None:
            self._completed_parabola_fit = refit

    def update_secondary(
        self,
        throw_label: int,
        motion_mask: np.ndarray | None,
        *,
        wrist_pos: tuple[int, int] | None = None,
        alternate_motion_mask: np.ndarray | None = None,
        defer_detecting_throw: bool = False,
        frame_index: int | None = None,
    ) -> TrajectoryResult:
        """
        Ball-only tracker for a secondary camera view.

        Throw detection comes from the main camera (``throw_label`` only). During
        scanning, uses ``wrist_pos`` from this camera's pose when available;
        otherwise falls back to a full-frame ball search.
        """
        if self.phase == Phase.AWAITING_PARTNER:
            self._awaiting_frame_count += 1
            if self._awaiting_frame_count >= self.awaiting_partner_timeout_frames:
                self.exit_awaiting_partner()
            return self._result_snapshot(None)

        detected_pos: tuple[int, int] | None = None

        if self.phase == Phase.DETECTING_THROW:
            if throw_label == 1:
                if wrist_pos is not None:
                    self._enter_scanning(wrist_pos)
                else:
                    self._enter_scanning_full_frame()

        elif self.phase == Phase.SCANNING_BALL:
            if throw_label == 1:
                self._scan_frame_count = 0
                if wrist_pos is not None:
                    self._scan_origin = wrist_pos
                    self._full_frame_scan = False

            if self._full_frame_scan or self._scan_origin is not None:
                detected_pos = self._find_ball_merged(motion_mask, alternate_motion_mask)
                if detected_pos is not None:
                    self._full_frame_scan = False
                    self._scan_origin = detected_pos
                    self._trajectory_points = [detected_pos]
                    self._trajectory_frames = (
                        [frame_index] if frame_index is not None else []
                    )
                    self._miss_count = 0
                    self._tracking_frame_count = 1
                    self.phase = Phase.TRACKING_BALL

            if self.phase == Phase.SCANNING_BALL and throw_label == 0:
                self._scan_frame_count += 1
                if self._scan_frame_count >= self.scanning_timeout_frames:
                    self._exit_scanning_failed(defer_detecting_throw)

        else:  # TRACKING_BALL
            self._tracking_frame_count += 1
            if throw_label == 1:
                saved = self._finalize_trajectory()
                if defer_detecting_throw:
                    if saved:
                        self._enter_awaiting_partner()
                    else:
                        self._mark_stereo_reconcile()
                else:
                    if wrist_pos is not None:
                        self._enter_scanning(wrist_pos)
                    else:
                        self._enter_scanning_full_frame()
            else:
                detected_pos = self._find_ball_merged(motion_mask, alternate_motion_mask)

                if detected_pos is not None:
                    if self._accept_tracking_detection(detected_pos, frame_index):
                        self._miss_count = 0
                    else:
                        self._miss_count += 1
                else:
                    self._miss_count += 1
                if self._miss_count >= self.timeout_frames:
                    saved = self._finalize_trajectory()
                    if defer_detecting_throw:
                        self._complete_throw_stereo(saved)
                    else:
                        self.phase = Phase.DETECTING_THROW

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
            completed_parabola_fit=self._completed_parabola_fit,
            completed_trajectory_frames=(
                list(self._completed_trajectory_frames)
                if self._completed_trajectory_frames is not None
                else None
            ),
            completed_tracking_frames=self._completed_tracking_frames,
            completion_id=self._completion_id,
        )

    def _enter_scanning(self, wrist_pos: tuple[int, int]) -> None:
        self.phase = Phase.SCANNING_BALL
        self._scan_origin = wrist_pos
        self._full_frame_scan = False
        self._trajectory_points = []
        self._trajectory_frames = []
        self._miss_count = 0
        self._scan_frame_count = 0

    def _enter_scanning_full_frame(self) -> None:
        self.phase = Phase.SCANNING_BALL
        self._scan_origin = None
        self._full_frame_scan = True
        self._trajectory_points = []
        self._trajectory_frames = []
        self._miss_count = 0
        self._scan_frame_count = 0

    def _enter_scanning_at_origin(self, origin: tuple[int, int]) -> None:
        self.phase = Phase.SCANNING_BALL
        self._scan_origin = origin
        self._full_frame_scan = False
        self._trajectory_points = []
        self._trajectory_frames = []
        self._miss_count = 0
        self._scan_frame_count = 0

    def _enter_awaiting_partner(self, *, valid_completion: bool = True) -> None:
        self.phase = Phase.AWAITING_PARTNER
        self._awaiting_frame_count = 0
        self._awaiting_valid_completion = valid_completion
        self._scan_origin = None
        self._full_frame_scan = False

    def _exit_scanning_failed(self, defer_detecting_throw: bool) -> None:
        self._scan_origin = None
        self._full_frame_scan = False
        if defer_detecting_throw:
            self._mark_stereo_reconcile()
        else:
            self.phase = Phase.DETECTING_THROW

    def _accept_tracking_detection(
        self,
        detected_pos: tuple[int, int],
        frame_index: int | None,
    ) -> bool:
        """Append a tracking hit, or reject it as a table-bounce miss."""
        self._scan_origin = detected_pos
        if self._is_bounce_motion_miss(detected_pos):
            return False
        self._trajectory_points.append(detected_pos)
        if frame_index is not None:
            self._trajectory_frames.append(frame_index)
        return True

    def _is_bounce_motion_miss(self, detected_pos: tuple[int, int]) -> bool:
        """
        After enough arc points, treat upward ball motion as a miss.

        Screen coordinates: +y is down. Upward world motion decreases y.
        A table bounce shows as upward velocity and/or upward acceleration.
        """
        points = self._trajectory_points
        if len(points) < BOUNCE_MISS_MIN_POINTS:
            return False

        _, last_y = points[-1]
        vy = detected_pos[1] - last_y
        if vy < 0:
            return True

        if len(points) >= 2:
            _, prev_y = points[-2]
            v_prev_y = last_y - prev_y
            if vy - v_prev_y < 0:
                return True

        return False

    def _finalize_trajectory(self) -> bool:
        """Fit a parabola to collected points and save the completed trajectory.

        Returns True when the trajectory was kept, False when discarded.
        """
        points = self._trajectory_points
        point_frames = self._trajectory_frames
        tracking_frames = self._tracking_frame_count
        self._trajectory_points = []
        self._trajectory_frames = []
        self._tracking_frame_count = 0

        if len(points) < MIN_TRAJECTORY_POINTS:
            return False

        self._completed_trajectory = list(points)
        self._completed_trajectory_frames = (
            list(point_frames) if len(point_frames) == len(points) else None
        )
        self._fitted_curve_points = None
        self._completed_parabola_fit = None

        fit = fit_parabola(points)
        if fit is not None:
            self._completed_parabola_fit = fit
            self._fitted_curve_points = sample_parabola(fit, points)

        self._completed_tracking_frames = tracking_frames
        self._completion_id += 1
        print(f"Completed trajectory: {self._completed_trajectory}")
        print(f"Completed trajectory frames: {self._completed_trajectory_frames}")
        print(f"Completed parabola fit: {self._completed_parabola_fit}")
        print(f"Completed tracking frames: {self._completed_tracking_frames}")
        print(f"Completion ID: {self._completion_id}")
        return True

    def _find_ball_merged(
        self,
        motion_mask: np.ndarray | None,
        alternate_motion_mask: np.ndarray | None = None,
    ) -> tuple[int, int] | None:
        """Prefer MOG2 (primary mask), then frame diff when both detect in the search area."""
        primary = self._find_ball(motion_mask) if motion_mask is not None else None
        if primary is not None or alternate_motion_mask is None:
            return primary
        return self._find_ball(alternate_motion_mask)

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
