from __future__ import annotations

import math

import cv2
import numpy as np

from .tracker import Phase, TrajectoryResult

# Sector outline
_SECTOR_COLOR_SCAN = (0, 200, 255)      # yellow-orange while scanning
_SECTOR_COLOR_TRACKING = (0, 255, 128)  # green while tracking
_SECTOR_THICKNESS = 3

# Ball detection marker
_BALL_MARKER_COLOR = (0, 100, 150)  # orange
_BALL_MARKER_RADIUS = 8
_BALL_MARKER_THICKNESS = 6

# Active trajectory points
_ACTIVE_PT_COLOR = (0, 255, 200)
_ACTIVE_PT_RADIUS = 5

# Completed trajectory points
_COMPLETED_PT_COLOR = (180, 180, 255)
_COMPLETED_PT_RADIUS = 4

# Fitted parabola curve
_CURVE_COLOR = (255, 80, 200)
_CURVE_THICKNESS = 5

# Phase label
_PHASE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PHASE_FONT_SCALE = 0.55
_PHASE_FONT_THICKNESS = 1
_LARGE_PHASE_FONT = cv2.FONT_HERSHEY_DUPLEX
_LARGE_PHASE_FONT_SCALE = 1.9
_LARGE_PHASE_FONT_THICKNESS = 3
_PHASE_BG = (0, 0, 0)
_PHASE_TEXT = (220, 220, 220)

# Speed readout (top-right, shown after trajectory completes)
_SPEED_FONT_SCALE = 1.35
_SPEED_FONT_THICKNESS = 2
_SPEED_BG = (0, 0, 0)
_SPEED_TEXT = (180, 255, 180)


def draw_sector(
    frame: np.ndarray,
    origin: tuple[int, int],
    direction_deg: float | None,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a circular-sector outline (two radii + arc) onto *frame* in-place."""
    ox, oy = origin
    if direction_deg is None:
        cv2.circle(frame, origin, radius, color, thickness)
        return

    half = 45.0  # fallback; real value is passed via TrajectoryTracker
    # The caller passes the tracker's half-angle through the wrapper below.
    # This bare function is only called from draw_trajectory_overlay.
    _draw_sector_impl(frame, ox, oy, direction_deg, half, radius, color, thickness)


def _draw_sector_impl(
    frame: np.ndarray,
    ox: int,
    oy: int,
    direction_deg: float,
    half_angle_deg: float,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    a1 = math.radians(direction_deg - half_angle_deg)
    a2 = math.radians(direction_deg + half_angle_deg)
    p1 = (int(ox + radius * math.cos(a1)), int(oy + radius * math.sin(a1)))
    p2 = (int(ox + radius * math.cos(a2)), int(oy + radius * math.sin(a2)))
    cv2.line(frame, (ox, oy), p1, color, thickness)
    cv2.line(frame, (ox, oy), p2, color, thickness)
    start_deg = direction_deg - half_angle_deg
    end_deg = direction_deg + half_angle_deg
    cv2.ellipse(frame, (ox, oy), (radius, radius), 0, start_deg, end_deg, color, thickness)


def draw_trajectory_overlay(
    frame: np.ndarray,
    result: TrajectoryResult,
    sector_half_angle_deg: float,
    sector_radius: int,
    *,
    speed_m_s: float | None = None,
    large_phase_label: bool = False,
) -> np.ndarray:
    """Draw all trajectory-tracking overlays on top of *frame* (copy returned)."""
    output = frame.copy()

    # Sector visualisation in phases 2 and 3.
    if result.phase in (Phase.SCANNING_BALL, Phase.TRACKING_BALL):
        if result.scan_origin is not None:
            color = (
                _SECTOR_COLOR_TRACKING
                if result.phase == Phase.TRACKING_BALL
                else _SECTOR_COLOR_SCAN
            )
            if result.scan_direction_deg is not None:
                _draw_sector_impl(
                    output,
                    result.scan_origin[0],
                    result.scan_origin[1],
                    result.scan_direction_deg,
                    sector_half_angle_deg,
                    sector_radius,
                    color,
                    _SECTOR_THICKNESS,
                )
            else:
                cv2.circle(output, result.scan_origin, sector_radius, color, _SECTOR_THICKNESS)

    # Current-frame ball detection marker.
    if result.detected_ball_pos is not None:
        cv2.circle(
            output,
            result.detected_ball_pos,
            _BALL_MARKER_RADIUS,
            _BALL_MARKER_COLOR,
            _BALL_MARKER_THICKNESS,
        )
        cv2.circle(output, result.detected_ball_pos, 2, _BALL_MARKER_COLOR, -1)

    # Active trajectory points (phase 3 in progress).
    for pt in result.trajectory_points:
        cv2.circle(output, pt, _ACTIVE_PT_RADIUS, _ACTIVE_PT_COLOR, -1)

    # Completed trajectory and curve persist until replaced by a valid finalize.
    if result.completed_trajectory:
        for pt in result.completed_trajectory:
            cv2.circle(output, pt, _COMPLETED_PT_RADIUS, _COMPLETED_PT_COLOR, -1)

        if result.fitted_curve_points and len(result.fitted_curve_points) >= 2:
            pts = np.array(result.fitted_curve_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [pts], False, _CURVE_COLOR, _CURVE_THICKNESS, cv2.LINE_AA)

    # Phase label (top-left).
    _draw_phase_label(output, result.phase, large=large_phase_label)

    # Speed label (top-right) for the last valid completed trajectory.
    if speed_m_s is not None and result.completed_trajectory:
        _draw_speed_label(output, speed_m_s)

    return output


def _draw_phase_label(frame: np.ndarray, phase: Phase, *, large: bool = False) -> None:
    text = phase.value.replace("_", " ")
    margin = 10
    if large:
        font = _LARGE_PHASE_FONT
        scale = _LARGE_PHASE_FONT_SCALE
        thickness = _LARGE_PHASE_FONT_THICKNESS
        pad_x, pad_y = 8, 6
    else:
        font = _PHASE_FONT
        scale = _PHASE_FONT_SCALE
        thickness = _PHASE_FONT_THICKNESS
        pad_x, pad_y = 4, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = margin, margin + th
    cv2.rectangle(
        frame,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + baseline + pad_y),
        _PHASE_BG,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        _PHASE_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def _draw_speed_label(frame: np.ndarray, speed_m_s: float) -> None:
    text = f"{speed_m_s:.1f} m/s  {speed_m_s * 3.6:.1f} km/h"
    margin = 10
    (tw, th), baseline = cv2.getTextSize(
        text, _PHASE_FONT, _SPEED_FONT_SCALE, _SPEED_FONT_THICKNESS
    )
    x = frame.shape[1] - margin - tw
    y = margin + th
    cv2.rectangle(
        frame,
        (x - 4, y - th - 2),
        (x + tw + 4, y + baseline + 2),
        _SPEED_BG,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        _PHASE_FONT,
        _SPEED_FONT_SCALE,
        _SPEED_TEXT,
        _SPEED_FONT_THICKNESS,
        cv2.LINE_AA,
    )
