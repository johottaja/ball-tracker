from __future__ import annotations

import cv2
import numpy as np

from .config import (
    BALL_CIRCULARITY_MAX,
    BALL_CIRCULARITY_MIN,
    DETECTION_RECT_THICKNESS,
)


def _to_gray(cleaned: np.ndarray) -> np.ndarray:
    if cleaned.ndim == 3:
        return cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    return cleaned


def _is_circular_contour(contour: np.ndarray) -> bool:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter**2)
    return BALL_CIRCULARITY_MIN < circularity <= BALL_CIRCULARITY_MAX


def find_largest_ball_contour(cleaned: np.ndarray) -> np.ndarray | None:
    gray = _to_gray(cleaned)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_ball_contours = [c for c in contours if _is_circular_contour(c)]
    if not valid_ball_contours:
        return None
    return max(valid_ball_contours, key=cv2.contourArea)


def draw_ball_contour(
    frame: np.ndarray,
    contour: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    output = frame.copy()
    if contour is not None:
        cv2.drawContours(output, [contour], -1, color, thickness)
    return output


def draw_ball_rectangle(
    frame: np.ndarray,
    contour: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = DETECTION_RECT_THICKNESS,
) -> np.ndarray:
    output = frame.copy()
    if contour is not None:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, thickness)
    return output
