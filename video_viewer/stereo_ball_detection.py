from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball_detection import (
    contour_bottom_center,
    find_hybrid_ball_contour,
    find_largest_ball_contour,
)
from .ball_motion import BallDetectionMethod, MotionMaskBuilder
from .filters import _mask_to_bgr


@dataclass(frozen=True)
class StreamBallDetection:
    motion_mask: np.ndarray | None
    alternate_motion_mask: np.ndarray | None
    ball_contour: np.ndarray | None
    ball_bottom: tuple[int, int] | None


@dataclass(frozen=True)
class StereoBallDetection:
    main: StreamBallDetection
    secondary: StreamBallDetection


def ball_contour_from_masks(
    method: BallDetectionMethod,
    motion_mask: np.ndarray | None,
    alternate_motion_mask: np.ndarray | None,
) -> np.ndarray | None:
    if method == BallDetectionMethod.HYBRID:
        return find_hybrid_ball_contour(motion_mask, alternate_motion_mask)
    if motion_mask is None:
        return None
    return find_largest_ball_contour(_mask_to_bgr(motion_mask))


def detect_stream_ball(
    motion_builder: MotionMaskBuilder,
    frame: np.ndarray,
    *,
    previous_frame: np.ndarray | None = None,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    cache: object | None = None,
    frame_index: int | None = None,
) -> StreamBallDetection:
    motion_mask, alternate_motion_mask = motion_builder.build_tracking_masks(
        frame,
        previous_frame,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        cache=cache,
        frame_index=frame_index,
    )
    contour = ball_contour_from_masks(
        motion_builder.method,
        motion_mask,
        alternate_motion_mask,
    )
    bottom = contour_bottom_center(contour) if contour is not None else None
    return StreamBallDetection(
        motion_mask=motion_mask,
        alternate_motion_mask=alternate_motion_mask,
        ball_contour=contour,
        ball_bottom=bottom,
    )


def detect_stereo_balls(
    main_motion: MotionMaskBuilder,
    secondary_motion: MotionMaskBuilder,
    main_frame: np.ndarray,
    secondary_frame: np.ndarray,
    *,
    main_previous_frame: np.ndarray | None = None,
    main_next_frame: np.ndarray | None = None,
    main_mog2_warmup_frames: list[np.ndarray] | None = None,
    secondary_previous_frame: np.ndarray | None = None,
    secondary_next_frame: np.ndarray | None = None,
    secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
    main_cache: object | None = None,
    secondary_cache: object | None = None,
    frame_index: int | None = None,
) -> StereoBallDetection:
    return StereoBallDetection(
        main=detect_stream_ball(
            main_motion,
            main_frame,
            previous_frame=main_previous_frame,
            next_frame=main_next_frame,
            mog2_warmup_frames=main_mog2_warmup_frames,
            cache=main_cache,
            frame_index=frame_index,
        ),
        secondary=detect_stream_ball(
            secondary_motion,
            secondary_frame,
            previous_frame=secondary_previous_frame,
            next_frame=secondary_next_frame,
            mog2_warmup_frames=secondary_mog2_warmup_frames,
            cache=secondary_cache,
            frame_index=frame_index,
        ),
    )
