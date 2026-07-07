from __future__ import annotations

import numpy as np

from framesync import FrameSyncEngine, draw_framesync_overlay
from framesync.playback import prepare_framesync_for_frame, record_framesync_completion
from throw_detection.inference import ThrowInference
from trajectory_tracking import TrajectoryTracker
from trajectory_tracking.stereo import reconcile_stereo_trackers
from trajectory_tracking.drawing import draw_trajectory_overlay
from trajectory_tracking.speed import TorsoLengthBuffer, estimate_throw_speed_m_s
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.filters import (
    FilterId,
    _draw_missing_model_banner,
    _extract_torso_length_px,
    _extract_wrist_pos,
)
from video_viewer.pose_overlay import apply_gru_throw_inference
from video_viewer.stereo_ball_detection import detect_stereo_balls


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
        self._framesync_engine = FrameSyncEngine()
        self._torso_length_buffer = TorsoLengthBuffer()
        self._main_completed_speed_m_s: float | None = None
        self._secondary_completed_speed_m_s: float | None = None
        self._main_last_completion_id: int = 0
        self._secondary_last_completion_id: int = 0
        self._last_frame_index: int | None = None

    @property
    def framesync_offset(self) -> float | None:
        return self._framesync_engine.latest_offset

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
        self._framesync_engine.reset()
        self._torso_length_buffer.reset()
        self._main_completed_speed_m_s = None
        self._secondary_completed_speed_m_s = None
        self._main_last_completion_id = 0
        self._secondary_last_completion_id = 0
        self._last_frame_index = None

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

    def apply(
        self,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
        *,
        frame_index: int | None = None,
        main_warmup_frames: list[np.ndarray] | None = None,
        main_warmup_start_index: int | None = None,
        main_previous_frame: np.ndarray | None = None,
        main_next_frame: np.ndarray | None = None,
        main_mog2_warmup_frames: list[np.ndarray] | None = None,
        secondary_previous_frame: np.ndarray | None = None,
        secondary_next_frame: np.ndarray | None = None,
        secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
        cache: object | None = None,
        ball_method: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            cache is not None
            and frame_index is not None
            and ball_method is not None
            and cache.has_stereo_output(
                FilterId.STEREO_TRACKING.value, ball_method, frame_index
            )
        ):
            return cache.get_stereo_output(
                FilterId.STEREO_TRACKING.value, ball_method, frame_index
            )

        inference = self._ensure_throw_inference()

        if main_warmup_frames is not None:
            self._main_tracker.reset()
            self._secondary_tracker.reset()
            self._main_motion.reset()
            self._secondary_motion.reset()
            self._framesync_engine.reset()
            self._torso_length_buffer.reset()
            self._main_completed_speed_m_s = None
            self._secondary_completed_speed_m_s = None
            self._main_last_completion_id = 0
            self._secondary_last_completion_id = 0
            self._last_frame_index = None

        if frame_index is not None:
            self._last_frame_index = prepare_framesync_for_frame(
                self._framesync_engine,
                frame_index,
                self._last_frame_index,
                cache,
            )

        detection = detect_stereo_balls(
            self._main_motion,
            self._secondary_motion,
            main_frame,
            secondary_frame,
            main_previous_frame=main_previous_frame,
            main_next_frame=main_next_frame,
            main_mog2_warmup_frames=main_mog2_warmup_frames,
            secondary_previous_frame=secondary_previous_frame,
            secondary_next_frame=secondary_next_frame,
            secondary_mog2_warmup_frames=secondary_mog2_warmup_frames,
            main_cache=cache.main if cache is not None else None,
            secondary_cache=cache.secondary if cache is not None else None,
            frame_index=frame_index,
        )

        sync_id_before = self._framesync_engine.sync_id
        framesync_result = None
        if frame_index is not None:
            framesync_result = self._framesync_engine.update(
                frame_index,
                detection.main.ball_bottom,
                detection.secondary.ball_bottom,
                video_fps=video_fps,
            )
            record_framesync_completion(
                self._framesync_engine,
                frame_index,
                sync_id_before,
                cache,
            )

        if inference is None:
            from video_viewer.pose_overlay import apply_normalized_throw_detection

            main_output = _draw_missing_model_banner(
                apply_normalized_throw_detection(
                    main_frame,
                    cache=cache.main if cache is not None else None,
                    frame_index=frame_index,
                )
            )
            main_output = self._apply_framesync_overlay(
                main_output,
                framesync_result,
                is_main=True,
            )
            secondary_output = self._apply_framesync_overlay(
                secondary_frame.copy(),
                framesync_result,
                is_main=False,
            )
            if (
                cache is not None
                and frame_index is not None
                and ball_method is not None
            ):
                cache.put_stereo_output(
                    FilterId.STEREO_TRACKING.value,
                    ball_method,
                    frame_index,
                    main_output,
                    secondary_output,
                )
            return main_output, secondary_output

        prediction = inference.predict(
            main_frame,
            warmup_frames=main_warmup_frames,
            warmup_start_index=main_warmup_start_index,
            cache=cache.main if cache is not None else None,
            frame_index=frame_index,
        )
        self._torso_length_buffer.add(_extract_torso_length_px(prediction.detection))
        throw_label = prediction.label

        main_result = self._main_tracker.update(
            throw_label=throw_label,
            wrist_pos=_extract_wrist_pos(prediction.detection),
            motion_mask=detection.main.motion_mask,
            alternate_motion_mask=detection.main.alternate_motion_mask,
            defer_detecting_throw=True,
            frame_index=frame_index,
        )
        secondary_result = self._secondary_tracker.update_secondary(
            throw_label=throw_label,
            motion_mask=detection.secondary.motion_mask,
            alternate_motion_mask=detection.secondary.alternate_motion_mask,
            defer_detecting_throw=True,
            frame_index=frame_index,
        )
        reconcile_stereo_trackers(
            self._main_tracker,
            self._secondary_tracker,
            throw_label=throw_label,
            wrist_pos=_extract_wrist_pos(prediction.detection),
        )

        if main_result.completion_id != self._main_last_completion_id:
            self._main_tracker.apply_release_extension(
                cache.main if cache is not None else None
            )
        if secondary_result.completion_id != self._secondary_last_completion_id:
            release_frames = self._main_tracker._completed_trajectory_frames
            if release_frames:
                self._secondary_tracker.apply_secondary_release_extension(release_frames[0])

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
            large_phase_label=True,
        )
        main_output = self._apply_framesync_overlay(
            main_output,
            framesync_result,
            is_main=True,
        )
        secondary_output = draw_trajectory_overlay(
            secondary_frame,
            secondary_result,
            self._secondary_tracker._sector_half_angle,
            self._secondary_tracker.sector_radius,
            speed_m_s=self._secondary_completed_speed_m_s,
            large_phase_label=True,
        )
        secondary_output = self._apply_framesync_overlay(
            secondary_output,
            framesync_result,
            is_main=False,
        )
        if (
            cache is not None
            and frame_index is not None
            and ball_method is not None
        ):
            cache.put_stereo_output(
                FilterId.STEREO_TRACKING.value,
                ball_method,
                frame_index,
                main_output,
                secondary_output,
            )
        return main_output, secondary_output

    def _apply_framesync_overlay(
        self,
        frame: np.ndarray,
        framesync_result,
        *,
        is_main: bool,
    ) -> np.ndarray:
        if framesync_result is None:
            return frame
        camera = framesync_result.main if is_main else framesync_result.secondary
        sync_display = (
            framesync_result.main_sync_display
            if is_main
            else framesync_result.secondary_sync_display
        )
        return draw_framesync_overlay(
            frame,
            phase=camera.phase,
            sync_display=sync_display,
            detected_ball_bottom=camera.detected_ball_bottom,
        )
