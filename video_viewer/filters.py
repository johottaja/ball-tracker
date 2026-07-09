from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from .ball_detection import (
    draw_ball_rectangle,
    draw_circular_contours,
    find_circular_contours,
    find_hybrid_ball_contour,
    find_hybrid_circular_contours,
    find_largest_ball_contour,
)
from .ball_motion import (
    BallDetectionMethod,
    MotionMaskBuilder,
    combine_hybrid_masks,
)
from throw_detection.inference import ThrowInference
from pose_detection import torso_scale
from trajectory_tracking import TrajectoryTracker
from trajectory_tracking.drawing import draw_trajectory_overlay
from trajectory_tracking.speed import TorsoLengthBuffer, estimate_throw_speed_m_s

from .pose_overlay import (
    apply_gru_throw_inference,
    apply_normalized_throw_detection,
    apply_throw_detection,
)
from .config import THROW_MODEL_PATH


class FilterId(str, Enum):
    NONE = "none"
    CONTOURS = "contours"
    DETECTION = "detection"
    THROW_DETECTION = "throw_detection"
    NORMALIZED_THROW_DETECTION = "normalized_throw_detection"
    GRU_THROW_INFERENCE = "gru_throw_inference"
    TRAJECTORY_TRACKING = "trajectory_tracking"
    STEREO_TRACKING = "stereo_tracking"
    FRAME_SYNC = "frame_sync"


STEREO_ONLY_FILTER_IDS = frozenset({FilterId.STEREO_TRACKING, FilterId.FRAME_SYNC})

FILTER_LABELS: dict[FilterId, str] = {
    FilterId.NONE: "None",
    FilterId.CONTOURS: "Contours",
    FilterId.DETECTION: "Ball detection",
    FilterId.THROW_DETECTION: "Throw detection",
    FilterId.NORMALIZED_THROW_DETECTION: "Normalized throw detection",
    FilterId.GRU_THROW_INFERENCE: "GRU throw inference",
    FilterId.TRAJECTORY_TRACKING: "Trajectory tracking",
    FilterId.STEREO_TRACKING: "Stereo tracking",
    FilterId.FRAME_SYNC: "Frame sync",
}

# Filters that build a motion mask from the video stream.
BALL_MASK_FILTER_IDS = frozenset(
    {
        FilterId.CONTOURS,
        FilterId.DETECTION,
        FilterId.TRAJECTORY_TRACKING,
        FilterId.STEREO_TRACKING,
        FilterId.FRAME_SYNC,
    }
)

_GRU_FILTER_IDS = frozenset({FilterId.GRU_THROW_INFERENCE, FilterId.TRAJECTORY_TRACKING})


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def detect_ball_contour(
    motion_builder: MotionMaskBuilder,
    current: np.ndarray,
    reference: np.ndarray | None,
    *,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    cache: object | None = None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    if motion_builder.method == BallDetectionMethod.HYBRID:
        mog2_mask, frame_diff_mask = motion_builder.build_component_masks(
            current,
            reference,
            next_frame=next_frame,
            mog2_warmup_frames=mog2_warmup_frames,
            cache=cache,
            frame_index=frame_index,
        )
        return find_hybrid_ball_contour(mog2_mask, frame_diff_mask)

    cleaned = motion_builder.build_mask(
        current,
        reference,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        cache=cache,
        frame_index=frame_index,
    )
    if cleaned is None:
        return None
    return find_largest_ball_contour(_mask_to_bgr(cleaned))


def apply_contours_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_bgr = _mask_to_bgr(mask)
    contours = find_circular_contours(mask_bgr)
    return draw_circular_contours(mask_bgr, contours)


def apply_hybrid_contours(
    mog2_mask: np.ndarray | None,
    frame_diff_mask: np.ndarray | None,
) -> np.ndarray:
    background = combine_hybrid_masks(mog2_mask, frame_diff_mask)
    if background is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    mask_bgr = _mask_to_bgr(background)
    contours = find_hybrid_circular_contours(mog2_mask, frame_diff_mask)
    return draw_circular_contours(mask_bgr, contours)


def apply_contours(
    motion_builder: MotionMaskBuilder,
    current: np.ndarray,
    reference: np.ndarray | None,
    *,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    cache: object | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    if motion_builder.method == BallDetectionMethod.HYBRID:
        mog2_mask, frame_diff_mask = motion_builder.build_component_masks(
            current,
            reference,
            next_frame=next_frame,
            mog2_warmup_frames=mog2_warmup_frames,
            cache=cache,
            frame_index=frame_index,
        )
        if mog2_mask is None and frame_diff_mask is None:
            return np.zeros_like(current)
        return apply_hybrid_contours(mog2_mask, frame_diff_mask)

    if (
        cache is not None
        and frame_index is not None
        and cache.has_motion_mask(motion_builder.method, frame_index)
    ):
        return apply_contours_from_mask(
            cache.get_motion_mask(motion_builder.method, frame_index)
        )

    cleaned = motion_builder.build_mask(
        current,
        reference,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        cache=cache,
        frame_index=frame_index,
    )
    if cleaned is None:
        return np.zeros_like(current)
    return apply_contours_from_mask(cleaned)


def apply_detection(
    motion_builder: MotionMaskBuilder,
    current: np.ndarray,
    reference: np.ndarray | None,
    *,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    cache: object | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    ball_contour = detect_ball_contour(
        motion_builder,
        current,
        reference,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        cache=cache,
        frame_index=frame_index,
    )
    return draw_ball_rectangle(current, ball_contour)


_STATEFUL_FILTER_IDS = frozenset(
    {FilterId.TRAJECTORY_TRACKING, FilterId.STEREO_TRACKING, FilterId.FRAME_SYNC}
)


class FrameFilter:
    """Applies the selected filter for on-screen display."""

    def __init__(self) -> None:
        self.filter_id = FilterId.NONE
        self._motion_builder = MotionMaskBuilder()
        self._throw_inference: ThrowInference | None = None
        self._trajectory_tracker: TrajectoryTracker | None = None
        self._torso_length_buffer = TorsoLengthBuffer()
        self._completed_speed_m_s: float | None = None
        self._last_completion_id: int = 0

    @property
    def ball_detection_method(self) -> BallDetectionMethod:
        return self._motion_builder.method

    def set_ball_detection_method(self, method: BallDetectionMethod) -> None:
        self._motion_builder.set_method(method)

    def reset(self) -> None:
        self._motion_builder.reset()
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

    def apply(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        next_frame: np.ndarray | None = None,
        mog2_warmup_frames: list[np.ndarray] | None = None,
        warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
        frame_index: int | None = None,
        cache: object | None = None,
        warmup_start_index: int | None = None,
    ) -> np.ndarray:
        if (
            cache is not None
            and frame_index is not None
            and self.filter_id in _STATEFUL_FILTER_IDS
            and cache.has_filter_output(
                self.filter_id.value, self.ball_detection_method, frame_index
            )
        ):
            return cache.get_filter_output(
                self.filter_id.value, self.ball_detection_method, frame_index
            )

        if self.filter_id == FilterId.NONE:
            return frame

        if self.filter_id == FilterId.CONTOURS:
            output = apply_contours(
                self._motion_builder,
                frame,
                previous_frame,
                next_frame=next_frame,
                mog2_warmup_frames=mog2_warmup_frames,
                cache=cache,
                frame_index=frame_index,
            )
        elif self.filter_id == FilterId.DETECTION:
            output = apply_detection(
                self._motion_builder,
                frame,
                previous_frame,
                next_frame=next_frame,
                mog2_warmup_frames=mog2_warmup_frames,
                cache=cache,
                frame_index=frame_index,
            )
        elif self.filter_id == FilterId.THROW_DETECTION:
            output = apply_throw_detection(
                frame,
                cache=cache,
                frame_index=frame_index,
            )
        elif self.filter_id == FilterId.NORMALIZED_THROW_DETECTION:
            output = apply_normalized_throw_detection(
                frame,
                cache=cache,
                frame_index=frame_index,
            )
        elif self.filter_id == FilterId.GRU_THROW_INFERENCE:
            output = self._apply_gru_throw_inference(
                frame,
                warmup_frames=warmup_frames,
                warmup_start_index=warmup_start_index,
                cache=cache,
                frame_index=frame_index,
            )
        elif self.filter_id == FilterId.TRAJECTORY_TRACKING:
            output = self._apply_trajectory_tracking(
                frame,
                previous_frame=previous_frame,
                next_frame=next_frame,
                mog2_warmup_frames=mog2_warmup_frames,
                warmup_frames=warmup_frames,
                warmup_start_index=warmup_start_index,
                video_fps=video_fps,
                cache=cache,
                frame_index=frame_index,
            )
        else:
            return frame

        if (
            cache is not None
            and frame_index is not None
            and self.filter_id in _STATEFUL_FILTER_IDS
        ):
            cache.put_filter_output(
                self.filter_id.value,
                self.ball_detection_method,
                frame_index,
                output,
            )
        return output

    def _apply_gru_throw_inference(
        self,
        frame: np.ndarray,
        *,
        warmup_frames: list[np.ndarray] | None,
        warmup_start_index: int | None,
        cache: object | None,
        frame_index: int | None,
    ) -> np.ndarray:
        inference = self._ensure_throw_inference()
        if inference is None:
            output = apply_normalized_throw_detection(
                frame,
                cache=cache,
                frame_index=frame_index,
            )
            return _draw_missing_model_banner(output)
        prediction = inference.predict(
            frame,
            warmup_frames=warmup_frames,
            warmup_start_index=warmup_start_index,
            cache=cache,
            frame_index=frame_index,
        )
        return apply_gru_throw_inference(frame, prediction)

    def _apply_trajectory_tracking(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None,
        next_frame: np.ndarray | None = None,
        mog2_warmup_frames: list[np.ndarray] | None,
        warmup_frames: list[np.ndarray] | None,
        warmup_start_index: int | None,
        video_fps: float | None,
        cache: object | None,
        frame_index: int | None,
    ) -> np.ndarray:
        inference = self._ensure_throw_inference()
        tracker = self._ensure_trajectory_tracker()

        if warmup_frames is not None:
            tracker.reset()
            self._motion_builder.reset()
            if mog2_warmup_frames is None and warmup_frames:
                self._motion_builder.warm_mog2(warmup_frames)
            self._torso_length_buffer.reset()
            self._completed_speed_m_s = None
            self._last_completion_id = 0

        if inference is None:
            self._motion_builder.build_tracking_masks(
                frame,
                previous_frame,
                next_frame=next_frame,
                cache=cache,
                frame_index=frame_index,
            )
            output = apply_normalized_throw_detection(
                frame,
                cache=cache,
                frame_index=frame_index,
            )
            return _draw_missing_model_banner(output)

        prediction = inference.predict(
            frame,
            warmup_frames=warmup_frames,
            warmup_start_index=warmup_start_index,
            cache=cache,
            frame_index=frame_index,
        )

        self._torso_length_buffer.add(_extract_torso_length_px(prediction.detection))

        motion_mask, alternate_motion_mask = self._motion_builder.build_tracking_masks(
            frame,
            previous_frame,
            next_frame=next_frame,
            mog2_warmup_frames=mog2_warmup_frames,
            cache=cache,
            frame_index=frame_index,
        )

        wrist_pos = _extract_wrist_pos(prediction.detection)
        tracking_result = tracker.update(
            throw_label=prediction.label,
            wrist_pos=wrist_pos,
            motion_mask=motion_mask,
            alternate_motion_mask=alternate_motion_mask,
            frame_index=frame_index,
        )

        if tracking_result.completion_id != self._last_completion_id:
            self._last_completion_id = tracking_result.completion_id
            tracker.apply_release_extension(cache)
            tracking_result = tracker._result_snapshot(tracking_result.detected_ball_pos)
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


def wrist_pos_from_frame(
    frame: np.ndarray,
    *,
    cache: object | None = None,
    frame_index: int | None = None,
) -> tuple[int, int] | None:
    """Dominant-hand wrist in frame pixels, using pose cache when available."""
    if cache is not None and frame_index is not None:
        from .playback_cache import cached_pose_detection

        detection = cached_pose_detection(
            frame,
            cache=cache,
            frame_index=frame_index,
        )
    else:
        from pose_detection import detect_dominant_hand_detection

        detection = detect_dominant_hand_detection(frame)
    return _extract_wrist_pos(detection)


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
