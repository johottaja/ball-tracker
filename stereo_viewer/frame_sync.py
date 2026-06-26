from __future__ import annotations

import numpy as np

from framesync import FrameSyncEngine, draw_framesync_overlay
from video_viewer.ball_detection import (
    contour_bottom_center,
    draw_ball_rectangle,
    find_largest_ball_contour,
)
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.filters import _mask_to_bgr


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
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._last_frame_index is not None
            and frame_index != self._last_frame_index + 1
        ):
            self._engine.reset()
            self._main_motion.reset()
            self._secondary_motion.reset()

        if main_mog2_warmup_frames is not None:
            self._main_motion.reset()
            if main_mog2_warmup_frames:
                self._main_motion.warm_mog2(main_mog2_warmup_frames)
        if secondary_mog2_warmup_frames is not None:
            self._secondary_motion.reset()
            if secondary_mog2_warmup_frames:
                self._secondary_motion.warm_mog2(secondary_mog2_warmup_frames)

        main_contour = self._detect_ball(
            self._main_motion,
            main_frame,
            main_previous_frame,
            mog2_warmup_frames=main_mog2_warmup_frames,
        )
        secondary_contour = self._detect_ball(
            self._secondary_motion,
            secondary_frame,
            secondary_previous_frame,
            mog2_warmup_frames=secondary_mog2_warmup_frames,
        )

        main_bottom = (
            contour_bottom_center(main_contour) if main_contour is not None else None
        )
        secondary_bottom = (
            contour_bottom_center(secondary_contour)
            if secondary_contour is not None
            else None
        )

        result = self._engine.update(
            frame_index,
            main_bottom,
            secondary_bottom,
            video_fps=video_fps,
        )
        self._last_frame_index = frame_index

        main_output = draw_ball_rectangle(main_frame, main_contour)
        main_output = draw_framesync_overlay(
            main_output,
            phase=result.main.phase,
            sync_display=result.main_sync_display,
            detected_ball_bottom=result.main.detected_ball_bottom,
        )

        secondary_output = draw_ball_rectangle(secondary_frame, secondary_contour)
        secondary_output = draw_framesync_overlay(
            secondary_output,
            phase=result.secondary.phase,
            sync_display=result.secondary_sync_display,
            detected_ball_bottom=result.secondary.detected_ball_bottom,
        )
        return main_output, secondary_output

    def _detect_ball(
        self,
        motion_builder: MotionMaskBuilder,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        *,
        mog2_warmup_frames: list[np.ndarray] | None,
    ) -> np.ndarray | None:
        mask = motion_builder.build_mask(
            frame,
            previous_frame,
            mog2_warmup_frames=mog2_warmup_frames,
        )
        if mask is None:
            return None
        return find_largest_ball_contour(_mask_to_bgr(mask))
