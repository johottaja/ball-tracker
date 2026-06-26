from __future__ import annotations

import numpy as np

from throw_detection.inference import ThrowInference
from trajectory_tracking import Phase, TrajectoryTracker
from trajectory_tracking.drawing import draw_trajectory_overlay
from trajectory_tracking.speed import TorsoLengthBuffer, estimate_throw_speed_m_s
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.filters import (
    _draw_missing_model_banner,
    _extract_torso_length_px,
    _extract_wrist_pos,
)
from video_viewer.pose_overlay import apply_gru_throw_inference


class StereoTrackingProcessor:
    """
    Main camera: pose + GRU throw detection + ball trajectory.
    Secondary camera: ball trajectory only, driven by the main throw label.
    """

    def __init__(self) -> None:
        self._throw_inference: ThrowInference | None = None
        self._main_tracker = TrajectoryTracker()
        self._secondary_tracker = TrajectoryTracker()
        self._main_motion = MotionMaskBuilder()
        self._secondary_motion = MotionMaskBuilder()
        self._torso_length_buffer = TorsoLengthBuffer()
        self._main_completed_speed_m_s: float | None = None
        self._secondary_completed_speed_m_s: float | None = None
        self._main_last_completion_id: int = 0
        self._secondary_last_completion_id: int = 0

    def set_ball_detection_method(self, method: BallDetectionMethod) -> None:
        self._main_motion.set_method(method)
        self._secondary_motion.set_method(method)

    def reset(self) -> None:
        if self._throw_inference is not None:
            self._throw_inference.reset()
        self._main_tracker.reset()
        self._secondary_tracker.reset()
        self._main_motion.reset()
        self._secondary_motion.reset()
        self._torso_length_buffer.reset()
        self._main_completed_speed_m_s = None
        self._secondary_completed_speed_m_s = None
        self._main_last_completion_id = 0
        self._secondary_last_completion_id = 0

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

    def _update_speed_on_completion(
        self,
        tracking_result,
        *,
        last_completion_id: int,
        video_fps: float | None,
    ) -> tuple[float | None, int]:
        if tracking_result.completion_id == last_completion_id:
            return None, last_completion_id
        speed = estimate_throw_speed_m_s(
            tracking_result.fitted_curve_points,
            self._torso_length_buffer.smoothed,
            tracking_result.completed_tracking_frames,
            video_fps,
        )
        return speed, tracking_result.completion_id

    def _sync_stereo_phases(self) -> None:
        """Return both trackers to idle only once each has finished its trajectory."""
        main = self._main_tracker
        secondary = self._secondary_tracker
        if (
            main.phase == Phase.AWAITING_PARTNER
            and secondary.phase == Phase.AWAITING_PARTNER
        ):
            main.phase = Phase.DETECTING_THROW
            secondary.phase = Phase.DETECTING_THROW

    def apply(
        self,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
        *,
        main_warmup_frames: list[np.ndarray] | None = None,
        main_previous_frame: np.ndarray | None = None,
        main_mog2_warmup_frames: list[np.ndarray] | None = None,
        secondary_previous_frame: np.ndarray | None = None,
        secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        inference = self._ensure_throw_inference()

        if main_warmup_frames is not None:
            self._main_tracker.reset()
            self._secondary_tracker.reset()
            self._main_motion.reset()
            self._secondary_motion.reset()
            self._torso_length_buffer.reset()
            self._main_completed_speed_m_s = None
            self._secondary_completed_speed_m_s = None
            self._main_last_completion_id = 0
            self._secondary_last_completion_id = 0

        if inference is None:
            from video_viewer.pose_overlay import apply_normalized_throw_detection

            main_output = _draw_missing_model_banner(
                apply_normalized_throw_detection(main_frame)
            )
            self._main_motion.build_mask(main_frame, main_previous_frame)
            self._secondary_motion.build_mask(
                secondary_frame, secondary_previous_frame
            )
            return main_output, secondary_frame.copy()

        prediction = inference.predict(main_frame, warmup_frames=main_warmup_frames)
        self._torso_length_buffer.add(_extract_torso_length_px(prediction.detection))
        throw_label = prediction.label

        main_motion_mask = self._main_motion.build_mask(
            main_frame,
            main_previous_frame,
            mog2_warmup_frames=main_mog2_warmup_frames,
        )
        secondary_motion_mask = self._secondary_motion.build_mask(
            secondary_frame,
            secondary_previous_frame,
            mog2_warmup_frames=secondary_mog2_warmup_frames,
        )

        main_result = self._main_tracker.update(
            throw_label=throw_label,
            wrist_pos=_extract_wrist_pos(prediction.detection),
            motion_mask=main_motion_mask,
            defer_detecting_throw=True,
        )
        secondary_result = self._secondary_tracker.update_secondary(
            throw_label=throw_label,
            motion_mask=secondary_motion_mask,
            defer_detecting_throw=True,
        )
        self._sync_stereo_phases()
        main_result = self._main_tracker._result_snapshot(
            main_result.detected_ball_pos
        )
        secondary_result = self._secondary_tracker._result_snapshot(
            secondary_result.detected_ball_pos
        )

        main_speed, self._main_last_completion_id = self._update_speed_on_completion(
            main_result,
            last_completion_id=self._main_last_completion_id,
            video_fps=video_fps,
        )
        if main_speed is not None:
            self._main_completed_speed_m_s = main_speed

        secondary_speed, self._secondary_last_completion_id = (
            self._update_speed_on_completion(
                secondary_result,
                last_completion_id=self._secondary_last_completion_id,
                video_fps=video_fps,
            )
        )
        if secondary_speed is not None:
            self._secondary_completed_speed_m_s = secondary_speed

        main_output = apply_gru_throw_inference(main_frame, prediction)
        main_output = draw_trajectory_overlay(
            main_output,
            main_result,
            self._main_tracker._sector_half_angle,
            self._main_tracker.sector_radius,
            speed_m_s=self._main_completed_speed_m_s,
        )
        secondary_output = draw_trajectory_overlay(
            secondary_frame,
            secondary_result,
            self._secondary_tracker._sector_half_angle,
            self._secondary_tracker.sector_radius,
            speed_m_s=self._secondary_completed_speed_m_s,
        )
        return main_output, secondary_output
