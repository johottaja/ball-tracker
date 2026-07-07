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
    HYBRID = "hybrid"
    HYBRID_STACKED = "hybrid_stacked"


BALL_DETECTION_METHOD_LABELS: dict[BallDetectionMethod, str] = {
    BallDetectionMethod.MOG2_CLOSING: "MOG2 + morphological closing",
    BallDetectionMethod.FRAME_DIFF: "Frame diff",
    BallDetectionMethod.HYBRID: "Hybrid",
    BallDetectionMethod.HYBRID_STACKED: "Hybrid stacked",
}

HYBRID_METHODS = frozenset(
    {BallDetectionMethod.HYBRID, BallDetectionMethod.HYBRID_STACKED}
)


def uses_mog2_component(method: BallDetectionMethod) -> bool:
    return method in (
        BallDetectionMethod.MOG2_CLOSING,
        *HYBRID_METHODS,
    )


def uses_frame_diff_component(method: BallDetectionMethod) -> bool:
    return method in (BallDetectionMethod.FRAME_DIFF, *HYBRID_METHODS)


def _single_diff_thresh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(a, b)
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


def _frame_diff_mask(
    current: np.ndarray,
    reference: np.ndarray | None,
    next_frame: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Motion mask from frame differencing.

    Two-frame diff (``current`` vs ``reference``) is symmetric, so it lights up
    both where the object newly appeared *and* where it used to be — a fast-moving
    ball shows up as two separate blobs ("ghosting"). When ``next_frame`` is
    available (playback, where frames are seekable), three-frame differencing
    ANDs the backward diff with the forward diff (``next_frame`` vs ``current``):
    the object's current position is present in both diffs, while the previous-
    and next-position ghosts each only appear in one, so the AND cancels them.
    """
    if reference is None or reference.shape != current.shape:
        return None
    thresh_prev = _single_diff_thresh(current, reference)
    if next_frame is None or next_frame.shape != current.shape:
        return thresh_prev
    thresh_next = _single_diff_thresh(next_frame, current)
    combined = cv2.bitwise_and(thresh_prev, thresh_next)
    kernel = np.ones(
        (FRAME_DIFF_MORPH_KERNEL_SIZE, FRAME_DIFF_MORPH_KERNEL_SIZE),
        np.uint8,
    )
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


def combine_hybrid_masks(
    mog2_mask: np.ndarray | None,
    frame_diff_mask: np.ndarray | None,
) -> np.ndarray | None:
    if mog2_mask is None and frame_diff_mask is None:
        return None
    if mog2_mask is None:
        return frame_diff_mask
    if frame_diff_mask is None:
        return mog2_mask
    return cv2.bitwise_or(mog2_mask, frame_diff_mask)


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
        if not uses_mog2_component(self._method):
            return
        for frame in frames:
            self._mog2_mask(frame)

    def build_component_masks(
        self,
        current: np.ndarray,
        reference: np.ndarray | None = None,
        *,
        next_frame: np.ndarray | None = None,
        mog2_warmup_frames: list[np.ndarray] | None = None,
        cache: object | None = None,
        frame_index: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (mog2_mask, frame_diff_mask), caching each component separately."""
        if mog2_warmup_frames is not None:
            self.reset()
            self.warm_mog2(mog2_warmup_frames)

        mog2_mask: np.ndarray | None = None
        frame_diff_mask: np.ndarray | None = None

        if (
            cache is not None
            and frame_index is not None
            and cache.has_motion_mask(BallDetectionMethod.MOG2_CLOSING, frame_index)
        ):
            mog2_mask = cache.get_motion_mask(
                BallDetectionMethod.MOG2_CLOSING, frame_index
            )
        else:
            mog2_mask = self._mog2_mask(current)
            if cache is not None and frame_index is not None:
                cache.put_motion_mask(
                    BallDetectionMethod.MOG2_CLOSING, frame_index, mog2_mask
                )

        if (
            cache is not None
            and frame_index is not None
            and cache.has_motion_mask(BallDetectionMethod.FRAME_DIFF, frame_index)
        ):
            frame_diff_mask = cache.get_motion_mask(
                BallDetectionMethod.FRAME_DIFF, frame_index
            )
        else:
            prev = reference if reference is not None else self._prev_frame
            frame_diff_mask = _frame_diff_mask(current, prev, next_frame)
            if frame_diff_mask is not None and cache is not None and frame_index is not None:
                cache.put_motion_mask(
                    BallDetectionMethod.FRAME_DIFF, frame_index, frame_diff_mask
                )

        self._prev_frame = current.copy()
        return mog2_mask, frame_diff_mask

    def build_tracking_masks(
        self,
        current: np.ndarray,
        reference: np.ndarray | None = None,
        *,
        next_frame: np.ndarray | None = None,
        mog2_warmup_frames: list[np.ndarray] | None = None,
        cache: object | None = None,
        frame_index: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return (motion_mask, alternate_motion_mask) for trajectory tracking.

        Hybrid merges MOG2 first, then frame diff. Hybrid stacked ORs masks
        into a single motion_mask with no alternate.
        """
        if self._method == BallDetectionMethod.HYBRID:
            mog2_mask, frame_diff_mask = self.build_component_masks(
                current,
                reference,
                next_frame=next_frame,
                mog2_warmup_frames=mog2_warmup_frames,
                cache=cache,
                frame_index=frame_index,
            )
            return mog2_mask, frame_diff_mask

        mask = self.build_mask(
            current,
            reference,
            next_frame=next_frame,
            mog2_warmup_frames=mog2_warmup_frames,
            cache=cache,
            frame_index=frame_index,
        )
        return mask, None

    def build_mask(
        self,
        current: np.ndarray,
        reference: np.ndarray | None = None,
        *,
        next_frame: np.ndarray | None = None,
        mog2_warmup_frames: list[np.ndarray] | None = None,
        cache: object | None = None,
        frame_index: int | None = None,
    ) -> np.ndarray | None:
        if (
            cache is not None
            and frame_index is not None
            and cache.has_motion_mask(self._method, frame_index)
        ):
            mask = cache.get_motion_mask(self._method, frame_index)
            self._prev_frame = current.copy()
            return mask

        if mog2_warmup_frames is not None:
            self.reset()
            self.warm_mog2(mog2_warmup_frames)

        if self._method == BallDetectionMethod.FRAME_DIFF:
            prev = reference if reference is not None else self._prev_frame
            mask = _frame_diff_mask(current, prev, next_frame)
            self._prev_frame = current.copy()
        elif self._method == BallDetectionMethod.HYBRID_STACKED:
            mog2_mask, frame_diff_mask = self.build_component_masks(
                current,
                reference,
                next_frame=next_frame,
                cache=cache,
                frame_index=frame_index,
            )
            mask = combine_hybrid_masks(mog2_mask, frame_diff_mask)
            if mask is not None and cache is not None and frame_index is not None:
                cache.put_motion_mask(self._method, frame_index, mask)
            return mask
        elif self._method == BallDetectionMethod.HYBRID:
            self.build_component_masks(
                current,
                reference,
                next_frame=next_frame,
                cache=cache,
                frame_index=frame_index,
            )
            return None
        else:
            mask = self._mog2_mask(current)
            self._prev_frame = current.copy()

        if mask is not None and cache is not None and frame_index is not None:
            cache.put_motion_mask(self._method, frame_index, mask)
        return mask
