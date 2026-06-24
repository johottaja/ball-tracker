from __future__ import annotations

from collections import deque
from enum import Enum

import cv2
import numpy as np

from .ball_detection import (
    draw_ball_contour,
    draw_ball_rectangle,
    draw_circular_contours,
    find_circular_contours,
    find_largest_ball_contour,
)
from throw_detection.inference import ThrowInference
from pose_detection import torso_scale
from trajectory_tracking import TrajectoryTracker
from trajectory_tracking.config import SECTOR_ANGLE_DEG, SECTOR_RADIUS_PX
from trajectory_tracking.drawing import draw_trajectory_overlay
from trajectory_tracking.speed import TorsoLengthBuffer, estimate_throw_speed_m_s

from .pose_overlay import (
    apply_gru_throw_inference,
    apply_normalized_throw_detection,
    apply_throw_detection,
)
from .config import (
    DIFF_BRIGHTNESS_FACTOR,
    DIFF_THRESH_VALUE,
    FRAME_WINDOW_SIZE,
    MORPH_KERNEL_SIZE,
    THROW_MODEL_PATH,
)

class FilterId(str, Enum):
    NONE = "none"
    GRAYSCALE = "grayscale"
    FRAME_DIFF_ONLY = "frame_diff_only"
    FRAME_DIFF_BRIGHTNESS = "frame_diff_brightness"
    FRAME_DIFF_CLEANED = "frame_diff_cleaned"
    FRAME_DIFF_CONTOURS = "frame_diff_contours"
    FRAME_DIFF_WINDOW = "frame_diff_window"
    DETECTION = "detection"
    THROW_DETECTION = "throw_detection"
    NORMALIZED_THROW_DETECTION = "normalized_throw_detection"
    GRU_THROW_INFERENCE = "gru_throw_inference"
    TRAJECTORY_TRACKING = "trajectory_tracking"
    STEREO_TRACKING = "stereo_tracking"


STEREO_ONLY_FILTER_IDS = frozenset({FilterId.STEREO_TRACKING})

FILTER_LABELS: dict[FilterId, str] = {
    FilterId.NONE: "None",
    FilterId.GRAYSCALE: "Grayscale",
    FilterId.FRAME_DIFF_ONLY: "Diff (current − previous)",
    FilterId.FRAME_DIFF_BRIGHTNESS: "Diff + brightness",
    FilterId.FRAME_DIFF_CLEANED: "Diff + brightness + threshold + clean",
    FilterId.FRAME_DIFF_CONTOURS: "Diff + brightness + threshold + clean + contours",
    FilterId.FRAME_DIFF_WINDOW: (
        f"Diff window (current − mean of last {FRAME_WINDOW_SIZE})"
    ),
    FilterId.DETECTION: "Ball detection",
    FilterId.THROW_DETECTION: "Throw detection",
    FilterId.NORMALIZED_THROW_DETECTION: "Normalized throw detection",
    FilterId.GRU_THROW_INFERENCE: "GRU throw inference",
    FilterId.TRAJECTORY_TRACKING: "Trajectory tracking",
    FilterId.STEREO_TRACKING: "Stereo tracking",
}

# Filters that use the immediately previous frame as reference.
PREV_FRAME_DIFF_FILTER_IDS = frozenset(
    {
        FilterId.FRAME_DIFF_ONLY,
        FilterId.FRAME_DIFF_BRIGHTNESS,
        FilterId.FRAME_DIFF_CLEANED,
        FilterId.FRAME_DIFF_CONTOURS,
        FilterId.DETECTION,
    }
)


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


def compute_frame_diff(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    if reference is None or reference.shape != current.shape:
        return None
    return cv2.subtract(current, reference)


def amplify_diff(diff: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(diff, alpha=DIFF_BRIGHTNESS_FACTOR, beta=0)


def clean_threshold(thresh: np.ndarray) -> np.ndarray:
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def threshold_amplified(amplified: np.ndarray) -> np.ndarray:
    if amplified.ndim == 3:
        gray = cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY)
    else:
        gray = amplified
    _, thresh = cv2.threshold(gray, DIFF_THRESH_VALUE, 255, cv2.THRESH_BINARY)
    return clean_threshold(thresh)


def build_motion_mask(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    diff = compute_frame_diff(current, reference)
    if diff is None:
        return None
    return threshold_amplified(amplify_diff(diff))


def _empty_like(frame: np.ndarray) -> np.ndarray:
    return np.zeros_like(frame)


def _prev_frame_diff_stages(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    diff = compute_frame_diff(current, reference)
    if diff is None:
        return None
    amplified = amplify_diff(diff)
    cleaned = threshold_amplified(amplified)
    return amplified, cleaned


def apply_prev_frame_diff_only(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    diff = compute_frame_diff(current, reference)
    if diff is None:
        return _empty_like(current)
    return diff


def apply_prev_frame_diff_brightness(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    diff = compute_frame_diff(current, reference)
    if diff is None:
        return _empty_like(current)
    return amplify_diff(diff)


def apply_prev_frame_diff_cleaned(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    stages = _prev_frame_diff_stages(current, reference)
    if stages is None:
        return _empty_like(current)
    _, cleaned = stages
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def apply_prev_frame_diff_contours(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    stages = _prev_frame_diff_stages(current, reference)
    if stages is None:
        return _empty_like(current)
    _, cleaned = stages
    mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    contours = find_circular_contours(mask_bgr)
    return draw_circular_contours(mask_bgr, contours)


def apply_frame_diff_window(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    """Full mask + largest circular contour for the windowed reference."""
    if reference is None or reference.shape != current.shape:
        return _empty_like(current)
    cleaned = build_motion_mask(current, reference)
    mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    ball_contour = find_largest_ball_contour(mask_bgr)
    return draw_ball_contour(mask_bgr, ball_contour)


def detect_ball_contour(
    current: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    cleaned = build_motion_mask(current, reference)
    if cleaned is None:
        return None
    mask_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return find_largest_ball_contour(mask_bgr)


def apply_detection(current: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    ball_contour = detect_ball_contour(current, reference)
    return draw_ball_rectangle(current, ball_contour)


_PREV_FRAME_APPLIERS = {
    FilterId.FRAME_DIFF_ONLY: apply_prev_frame_diff_only,
    FilterId.FRAME_DIFF_BRIGHTNESS: apply_prev_frame_diff_brightness,
    FilterId.FRAME_DIFF_CLEANED: apply_prev_frame_diff_cleaned,
    FilterId.FRAME_DIFF_CONTOURS: apply_prev_frame_diff_contours,
    FilterId.DETECTION: apply_detection,
}


_GRU_FILTER_IDS = frozenset({FilterId.GRU_THROW_INFERENCE, FilterId.TRAJECTORY_TRACKING})


class FrameFilter:
    """Applies the selected filter for on-screen display."""

    def __init__(self, window_size: int = FRAME_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.filter_id = FilterId.NONE
        self._prev_frame: np.ndarray | None = None
        self._frame_window: deque[np.ndarray] = deque(maxlen=window_size)
        self._throw_inference: ThrowInference | None = None
        self._trajectory_tracker: TrajectoryTracker | None = None
        self._torso_length_buffer = TorsoLengthBuffer()
        self._completed_speed_m_s: float | None = None
        self._last_completion_id: int = 0

    def reset(self) -> None:
        self._prev_frame = None
        self._frame_window.clear()
        if self._throw_inference is not None:
            self._throw_inference.reset()
        if self._trajectory_tracker is not None:
            self._trajectory_tracker.reset()
        self._torso_length_buffer.reset()
        self._completed_speed_m_s = None
        self._last_completion_id = 0

    def set_filter(self, filter_id: FilterId) -> None:
        if filter_id != self.filter_id:
            self.filter_id = filter_id
            self.reset()
            if filter_id not in _GRU_FILTER_IDS:
                self._throw_inference = None
            if filter_id != FilterId.TRAJECTORY_TRACKING:
                self._trajectory_tracker = None
                self._torso_length_buffer.reset()
                self._completed_speed_m_s = None
                self._last_completion_id = 0

    def throw_buffer_size(self) -> int:
        inference = self._ensure_throw_inference()
        if inference is None:
            from throw_detection.config import BUFFER_SIZE

            return BUFFER_SIZE
        return inference.buffer_size

    def _ensure_throw_inference(self) -> ThrowInference | None:
        if self._throw_inference is not None:
            return self._throw_inference
        if THROW_MODEL_PATH is None or not THROW_MODEL_PATH.is_file():
            return None
        self._throw_inference = ThrowInference(THROW_MODEL_PATH)
        return self._throw_inference

    def _ensure_trajectory_tracker(self) -> TrajectoryTracker:
        if self._trajectory_tracker is None:
            self._trajectory_tracker = TrajectoryTracker()
        return self._trajectory_tracker

    def _apply_with_previous(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None,
        applier,
    ) -> np.ndarray:
        prev = previous_frame if previous_frame is not None else self._prev_frame
        result = applier(frame, prev)
        self._prev_frame = frame.copy()
        return result

    def apply(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        window_frames: list[np.ndarray] | None = None,
        warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
    ) -> np.ndarray:
        if self.filter_id == FilterId.NONE:
            return frame
        if self.filter_id == FilterId.GRAYSCALE:
            return apply_grayscale(frame)

        if self.filter_id in _PREV_FRAME_APPLIERS:
            return self._apply_with_previous(
                frame,
                previous_frame=previous_frame,
                applier=_PREV_FRAME_APPLIERS[self.filter_id],
            )

        if self.filter_id == FilterId.FRAME_DIFF_WINDOW:
            if window_frames is not None:
                reference = mean_frame(window_frames)
            else:
                reference = mean_frame(list(self._frame_window))
            result = apply_frame_diff_window(frame, reference)
            if window_frames is None:
                self._frame_window.append(frame.copy())
            return result

        if self.filter_id == FilterId.THROW_DETECTION:
            return apply_throw_detection(frame)

        if self.filter_id == FilterId.NORMALIZED_THROW_DETECTION:
            return apply_normalized_throw_detection(frame)

        if self.filter_id == FilterId.GRU_THROW_INFERENCE:
            inference = self._ensure_throw_inference()
            if inference is None:
                output = apply_normalized_throw_detection(frame)
                return _draw_missing_model_banner(output)
            prediction = inference.predict(frame, warmup_frames=warmup_frames)
            return apply_gru_throw_inference(frame, prediction)

        if self.filter_id == FilterId.TRAJECTORY_TRACKING:
            return self._apply_trajectory_tracking(
                frame,
                warmup_frames=warmup_frames,
                video_fps=video_fps,
            )

        return frame

    def _apply_trajectory_tracking(
        self,
        frame: np.ndarray,
        *,
        warmup_frames: list[np.ndarray] | None,
        video_fps: float | None,
    ) -> np.ndarray:
        inference = self._ensure_throw_inference()
        tracker = self._ensure_trajectory_tracker()

        if warmup_frames is not None:
            tracker.reset()
            self._prev_frame = warmup_frames[-1].copy() if warmup_frames else None
            self._torso_length_buffer.reset()
            self._completed_speed_m_s = None
            self._last_completion_id = 0

        if inference is None:
            self._prev_frame = frame.copy()
            output = apply_normalized_throw_detection(frame)
            return _draw_missing_model_banner(output)

        prediction = inference.predict(frame, warmup_frames=warmup_frames)

        self._torso_length_buffer.add(_extract_torso_length_px(prediction.detection))

        motion_mask = build_motion_mask(frame, self._prev_frame)
        self._prev_frame = frame.copy()

        wrist_pos = _extract_wrist_pos(prediction.detection)
        tracking_result = tracker.update(
            throw_label=prediction.label,
            wrist_pos=wrist_pos,
            motion_mask=motion_mask,
        )

        if tracking_result.completion_id != self._last_completion_id:
            self._last_completion_id = tracking_result.completion_id
            self._completed_speed_m_s = estimate_throw_speed_m_s(
                tracking_result.fitted_curve_points,
                self._torso_length_buffer.smoothed,
                tracking_result.completed_tracking_frames,
                video_fps,
            )

        output = apply_gru_throw_inference(frame, prediction)
        return draw_trajectory_overlay(
            output,
            tracking_result,
            tracker._sector_half_angle,
            tracker.sector_radius,
            speed_m_s=self._completed_speed_m_s,
        )


def _extract_wrist_pos(detection: object) -> tuple[int, int] | None:
    """Return the wrist pixel position from a detection, or None."""
    if detection is None:
        return None
    hand = getattr(detection, "hand", None)
    if hand is None:
        return None
    joints = hand.joints
    if len(joints) < 3:
        return None
    wrist = joints[2]
    return (int(wrist.x), int(wrist.y))


def _extract_torso_length_px(detection: object) -> float | None:
    """Return dominant-side shoulder-to-hip length in pixels, or None."""
    if detection is None:
        return None
    hand = getattr(detection, "hand", None)
    if hand is None:
        return None
    person_keypoints = getattr(detection, "person_keypoints", None)
    if person_keypoints is None:
        return None
    return torso_scale(person_keypoints, hand.side)


def _draw_missing_model_banner(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()
    line = "No GRU model in throw_detection/models/"
    cv2.putText(
        output,
        line,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output
