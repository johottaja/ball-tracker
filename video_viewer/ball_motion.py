from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from .config import (
    DIFF_BRIGHTNESS_FACTOR,
    DIFF_THRESH_VALUE,
    FRAME_DIFF_MORPH_KERNEL_SIZE,
    MOG2_DETECT_SHADOWS,
    MOG2_HISTORY,
    MOG2_MORPH_KERNEL_SIZE,
    MOG2_VAR_THRESHOLD,
)


class BallDetectionMethod(str, Enum):
    MOG2_CLOSING = "mog2_closing"
    FRAME_DIFF = "frame_diff"


BALL_DETECTION_METHOD_LABELS: dict[BallDetectionMethod, str] = {
    BallDetectionMethod.MOG2_CLOSING: "MOG2 + morphological closing",
    BallDetectionMethod.FRAME_DIFF: "Frame diff",
}


def _frame_diff_mask(current: np.ndarray, reference: np.ndarray | None) -> np.ndarray | None:
    if reference is None or reference.shape != current.shape:
        return None
    diff = cv2.subtract(current, reference)
    amplified = cv2.convertScaleAbs(diff, alpha=DIFF_BRIGHTNESS_FACTOR, beta=0)
    if amplified.ndim == 3:
        gray = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
    else:
        gray = amplified
    _, thresh = cv2.threshold(gray, DIFF_THRESH_VALUE, 255, cv2.THRESH_BINARY)
    kernel = np.ones(
        (FRAME_DIFF_MORPH_KERNEL_SIZE, FRAME_DIFF_MORPH_KERNEL_SIZE),
        np.uint8,
    )
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


class MotionMaskBuilder:
    """Builds a binary motion mask for ball contour detection."""

    def __init__(self, method: BallDetectionMethod = BallDetectionMethod.MOG2_CLOSING) -> None:
        self._method = method
        self._prev_frame: np.ndarray | None = None
        self._mog2: cv2.BackgroundSubtractorMOG2 | None = None

    @property
    def method(self) -> BallDetectionMethod:
        return self._method

    def set_method(self, method: BallDetectionMethod) -> None:
        if method != self._method:
            self._method = method
            self.reset()

    def reset(self) -> None:
        self._prev_frame = None
        self._mog2 = None

    def _ensure_mog2(self) -> cv2.BackgroundSubtractorMOG2:
        if self._mog2 is None:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY,
                varThreshold=MOG2_VAR_THRESHOLD,
                detectShadows=MOG2_DETECT_SHADOWS,
            )
        return self._mog2

    def _mog2_mask(self, frame: np.ndarray) -> np.ndarray:
        subtractor = self._ensure_mog2()
        fg_mask = subtractor.apply(frame)
        if MOG2_DETECT_SHADOWS:
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((MOG2_MORPH_KERNEL_SIZE, MOG2_MORPH_KERNEL_SIZE), np.uint8)
        return cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    def warm_mog2(self, frames: list[np.ndarray]) -> None:
        if self._method != BallDetectionMethod.MOG2_CLOSING:
            return
        for frame in frames:
            self._mog2_mask(frame)

    def build_mask(
        self,
        current: np.ndarray,
        reference: np.ndarray | None = None,
        *,
        mog2_warmup_frames: list[np.ndarray] | None = None,
    ) -> np.ndarray | None:
        if mog2_warmup_frames is not None:
            self.reset()
            self.warm_mog2(mog2_warmup_frames)

        if self._method == BallDetectionMethod.FRAME_DIFF:
            prev = reference if reference is not None else self._prev_frame
            mask = _frame_diff_mask(current, prev)
            self._prev_frame = current.copy()
            return mask

        mask = self._mog2_mask(current)
        self._prev_frame = current.copy()
        return mask
