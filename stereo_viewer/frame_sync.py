from __future__ import annotations

import numpy as np

from framesync import FrameSyncEngine, draw_framesync_overlay
from framesync.playback import prepare_framesync_for_frame, record_framesync_completion
from video_viewer.ball_detection import draw_ball_rectangle
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.filters import FilterId
from video_viewer.stereo_ball_detection import detect_stereo_balls


class FrameSyncProcessor:
    """Stereo frame-sync filter: ball drop/bounce detection on both cameras."""

    def __init__(self) -> None:
        self._engine = FrameSyncEngine()
        self._main_motion = MotionMaskBuilder()
        self._secondary_motion = MotionMaskBuilder()
        self._last_frame_index: int | None = None

    def set_ball_detection_method(self, method: BallDetectionMethod) -> None:
        self._main_motion.set_method(method)
        self._secondary_motion.set_method(method)

    def reset(self) -> None:
        self._engine.reset()
        self._main_motion.reset()
        self._secondary_motion.reset()
        self._last_frame_index = None

    def apply(
        self,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
        *,
        frame_index: int,
        main_previous_frame: np.ndarray | None = None,
        main_mog2_warmup_frames: list[np.ndarray] | None = None,
        secondary_previous_frame: np.ndarray | None = None,
        secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
        cache: object | None = None,
        ball_method: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            cache is not None
            and ball_method is not None
            and cache.has_stereo_output(
                FilterId.FRAME_SYNC.value, ball_method, frame_index
            )
        ):
            return cache.get_stereo_output(
                FilterId.FRAME_SYNC.value, ball_method, frame_index
            )

        if main_mog2_warmup_frames is not None:
            self._main_motion.reset()
            if main_mog2_warmup_frames:
                self._main_motion.warm_mog2(main_mog2_warmup_frames)
        if secondary_mog2_warmup_frames is not None:
            self._secondary_motion.reset()
            if secondary_mog2_warmup_frames:
                self._secondary_motion.warm_mog2(secondary_mog2_warmup_frames)

        self._last_frame_index = prepare_framesync_for_frame(
            self._engine,
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
            main_mog2_warmup_frames=main_mog2_warmup_frames,
            secondary_previous_frame=secondary_previous_frame,
            secondary_mog2_warmup_frames=secondary_mog2_warmup_frames,
            main_cache=cache.main if cache is not None else None,
            secondary_cache=cache.secondary if cache is not None else None,
            frame_index=frame_index,
        )

        sync_id_before = self._engine.sync_id
        result = self._engine.update(
            frame_index,
            detection.main.ball_bottom,
            detection.secondary.ball_bottom,
            video_fps=video_fps,
        )
        record_framesync_completion(
            self._engine,
            frame_index,
            sync_id_before,
            cache,
        )

        main_output = draw_ball_rectangle(main_frame, detection.main.ball_contour)
        main_output = draw_framesync_overlay(
            main_output,
            phase=result.main.phase,
            sync_display=result.main_sync_display,
            detected_ball_bottom=result.main.detected_ball_bottom,
        )

        secondary_output = draw_ball_rectangle(
            secondary_frame, detection.secondary.ball_contour
        )
        secondary_output = draw_framesync_overlay(
            secondary_output,
            phase=result.secondary.phase,
            sync_display=result.secondary_sync_display,
            detected_ball_bottom=result.secondary.detected_ball_bottom,
        )
        if cache is not None and ball_method is not None:
            cache.put_stereo_output(
                FilterId.FRAME_SYNC.value,
                ball_method,
                frame_index,
                main_output,
                secondary_output,
            )
        return main_output, secondary_output
