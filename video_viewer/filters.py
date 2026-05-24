from __future__ import annotations

from collections import deque
from enum import Enum

import cv2
import numpy as np

from .ball_detection import (
    draw_ball_contour,
    draw_ball_rectangle,
    find_largest_ball_contour,
)
from .config import (
    DIFF_BRIGHTNESS_FACTOR,
    DIFF_THRESH_VALUE,
    FRAME_WINDOW_SIZE,
    MORPH_KERNEL_SIZE,
)


class FilterId(str, Enum):
    NONE = "none"
    GRAYSCALE = "grayscale"
    FRAME_DIFF = "frame_diff"
    FRAME_DIFF_WINDOW = "frame_diff_window"
    DETECTION = "detection"


FILTER_LABELS: dict[FilterId, str] = {
    FilterId.NONE: "None",
    FilterId.GRAYSCALE: "Grayscale",
    FilterId.FRAME_DIFF: "Frame difference (current − previous)",
    FilterId.FRAME_DIFF_WINDOW: (
        f"Frame difference (current − mean of last {FRAME_WINDOW_SIZE})"
    ),
    FilterId.DETECTION: "Ball detection",
}


def apply_grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def mean_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.uint8)


def clean_threshold(thresh: np.ndarray) -> np.ndarray:
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def build_motion_mask(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    if reference is None or reference.shape != current.shape:
        return None
    diff = cv2.subtract(current, reference)
    amplified = cv2.convertScaleAbs(diff, alpha=DIFF_BRIGHTNESS_FACTOR, beta=0)
    if amplified.ndim == 3:
        gray = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
    else:
        gray = amplified
    _, thresh = cv2.threshold(gray, DIFF_THRESH_VALUE, 255, cv2.THRESH_BINARY)
    return clean_threshold(thresh)


def detect_ball_contour(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    cleaned = build_motion_mask(current, reference)
    if cleaned is None:
        return None
    mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return find_largest_ball_contour(mask_bgr)


def apply_frame_diff(current: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None or reference.shape != current.shape:
        return np.zeros_like(current)
    cleaned = build_motion_mask(current, reference)
    mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    ball_contour = find_largest_ball_contour(mask_bgr)
    return draw_ball_contour(mask_bgr, ball_contour)


def apply_detection(current: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    ball_contour = detect_ball_contour(current, reference)
    return draw_ball_rectangle(current, ball_contour)


class FrameFilter:
    """Applies the selected filter for on-screen display."""

    def __init__(self, window_size: int = FRAME_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.filter_id = FilterId.NONE
        self._prev_frame: np.ndarray | None = None
        self._frame_window: deque[np.ndarray] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._prev_frame = None
        self._frame_window.clear()

    def set_filter(self, filter_id: FilterId) -> None:
        if filter_id != self.filter_id:
            self.filter_id = filter_id
            self.reset()

    def apply(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        window_frames: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.filter_id == FilterId.NONE:
            return frame
        if self.filter_id == FilterId.GRAYSCALE:
            return apply_grayscale(frame)
        if self.filter_id == FilterId.FRAME_DIFF:
            prev = previous_frame if previous_frame is not None else self._prev_frame
            result = apply_frame_diff(frame, prev)
            self._prev_frame = frame.copy()
            return result
        if self.filter_id == FilterId.FRAME_DIFF_WINDOW:
            if window_frames is not None:
                reference = mean_frame(window_frames)
            else:
                reference = mean_frame(list(self._frame_window))
            result = apply_frame_diff(frame, reference)
            if window_frames is None:
                self._frame_window.append(frame.copy())
            return result
        if self.filter_id == FilterId.DETECTION:
            prev = previous_frame if previous_frame is not None else self._prev_frame
            result = apply_detection(frame, prev)
            self._prev_frame = frame.copy()
            return result
        return frame
