from __future__ import annotations

import cv2
import numpy as np

from .types import Phase

_SYNC_FONT = cv2.FONT_HERSHEY_DUPLEX
_SYNC_FONT_SCALE = 1.9
_SYNC_FONT_THICKNESS = 3
_SYNC_BG = (0, 0, 0)
_SYNC_TEXT = (200, 255, 255)

_PHASE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PHASE_FONT_SCALE = 0.55
_PHASE_FONT_THICKNESS = 1
_PHASE_BG = (0, 0, 0)
_PHASE_TEXT = (220, 220, 220)

_BALL_MARKER_COLOR = (0, 180, 255)
_BALL_MARKER_RADIUS = 6


def draw_framesync_overlay(
    frame: np.ndarray,
    *,
    phase: Phase,
    sync_display: float | None,
    detected_ball_bottom: tuple[int, int] | None,
) -> np.ndarray:
    output = frame.copy()

    if detected_ball_bottom is not None and phase != Phase.WATCHING:
        cv2.circle(
            output,
            detected_ball_bottom,
            _BALL_MARKER_RADIUS,
            _BALL_MARKER_COLOR,
            -1,
        )

    _draw_phase_label(output, phase)
    _draw_sync_label(output, sync_display)
    return output


def _draw_sync_label(frame: np.ndarray, sync_display: float | None) -> None:
    text = f"{sync_display:+.2f}" if sync_display is not None else "--"
    (tw, th), baseline = cv2.getTextSize(
        text,
        _SYNC_FONT,
        _SYNC_FONT_SCALE,
        _SYNC_FONT_THICKNESS,
    )
    x = (frame.shape[1] - tw) // 2
    y = 16 + th
    cv2.rectangle(
        frame,
        (x - 8, y - th - 6),
        (x + tw + 8, y + baseline + 6),
        _SYNC_BG,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        _SYNC_FONT,
        _SYNC_FONT_SCALE,
        _SYNC_TEXT,
        _SYNC_FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_phase_label(frame: np.ndarray, phase: Phase) -> None:
    text = phase.value
    margin = 10
    (tw, th), baseline = cv2.getTextSize(
        text,
        _PHASE_FONT,
        _PHASE_FONT_SCALE,
        _PHASE_FONT_THICKNESS,
    )
    x, y = margin, margin + th
    cv2.rectangle(
        frame,
        (x - 4, y - th - 2),
        (x + tw + 4, y + baseline + 2),
        _PHASE_BG,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        _PHASE_FONT,
        _PHASE_FONT_SCALE,
        _PHASE_TEXT,
        _PHASE_FONT_THICKNESS,
        cv2.LINE_AA,
    )
