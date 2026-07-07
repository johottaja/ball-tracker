from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageTk

from .ball_motion import (
    BallDetectionMethod,
    HYBRID_METHODS,
    uses_frame_diff_component,
    uses_mog2_component,
)
from .config import MOG2_HISTORY
from .display import frame_to_photo
from .filters import BALL_MASK_FILTER_IDS, FilterId, FrameFilter


def step_index_by_seconds(
    frame_index: int,
    fps: float,
    seconds: float,
    *,
    forward: bool = True,
    frame_count: int = 0,
) -> int:
    effective_fps = fps if fps > 0 else 30.0
    delta = max(1, round(effective_fps * abs(seconds)))
    new_index = frame_index + delta if forward else frame_index - delta
    new_index = max(0, new_index)
    if frame_count > 0:
        new_index = min(new_index, frame_count - 1)
    return new_index


def read_frame_at(cap: cv2.VideoCapture, index: int) -> tuple[bool, np.ndarray | None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return ok, frame if ok else None


def previous_frame_for_diff(
    cap: cv2.VideoCapture, index: int
) -> np.ndarray | None:
    if index <= 0:
        return None
    ok, frame = read_frame_at(cap, index - 1)
    return frame if ok else None


def next_frame_for_diff(
    cap: cv2.VideoCapture, index: int, frame_count: int = 0
) -> np.ndarray | None:
    """Read frame N+1 for three-frame differencing (playback is seekable)."""
    if frame_count > 0 and index + 1 >= frame_count:
        return None
    ok, frame = read_frame_at(cap, index + 1)
    return frame if ok else None


def mog2_warmup_frames(
    cap: cv2.VideoCapture, index: int, history: int = MOG2_HISTORY
) -> list[np.ndarray]:
    if index <= 0:
        return []
    start = max(0, index - history)
    frames: list[np.ndarray] = []
    for frame_index in range(start, index):
        ok, frame = read_frame_at(cap, frame_index)
        if ok and frame is not None:
            frames.append(frame)
    return frames


def mog2_warmup_frames_if_needed(
    cap: cv2.VideoCapture,
    index: int,
    mog2_stream_frame_index: int | None,
    history: int = MOG2_HISTORY,
) -> list[np.ndarray] | None:
    if mog2_stream_frame_index is not None and index == mog2_stream_frame_index + 1:
        return None
    return mog2_warmup_frames(cap, index, history)


def warmup_frames_for_gru(
    cap: cv2.VideoCapture, index: int, buffer_size: int
) -> list[np.ndarray]:
    if index <= 0:
        return []
    start = max(0, index - buffer_size + 1)
    warmup_frames: list[np.ndarray] = []
    for frame_index in range(start, index):
        ok, warmup_frame = read_frame_at(cap, frame_index)
        if ok and warmup_frame is not None:
            warmup_frames.append(warmup_frame)
    return warmup_frames


def gru_warmup_frames_if_needed(
    cap: cv2.VideoCapture,
    index: int,
    gru_stream_frame_index: int | None,
    buffer_size: int,
) -> list[np.ndarray] | None:
    if gru_stream_frame_index is not None and index == gru_stream_frame_index + 1:
        return None
    return warmup_frames_for_gru(cap, index, buffer_size)


def _frame_diff_mask_cached(cache: object | None, index: int) -> bool:
    return (
        cache is not None
        and cache.has_motion_mask(BallDetectionMethod.FRAME_DIFF, index)
    )


def _mog2_mask_cached(cache: object | None, index: int) -> bool:
    return (
        cache is not None
        and cache.has_motion_mask(BallDetectionMethod.MOG2_CLOSING, index)
    )


def ball_mask_playback_inputs(
    cap: cv2.VideoCapture,
    method: BallDetectionMethod,
    index: int,
    mog2_stream_frame_index: int | None,
    cache: object | None = None,
    frame_count: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
    previous: np.ndarray | None = None
    next_frame: np.ndarray | None = None
    mog2_warmup: list[np.ndarray] | None = None

    if uses_frame_diff_component(method) and not _frame_diff_mask_cached(cache, index):
        previous = previous_frame_for_diff(cap, index)
        next_frame = next_frame_for_diff(cap, index, frame_count)

    if method == BallDetectionMethod.MOG2_CLOSING:
        if cache is None or not cache.has_motion_mask(method, index):
            mog2_warmup = mog2_warmup_frames_if_needed(
                cap, index, mog2_stream_frame_index
            )
    elif method in HYBRID_METHODS and not _mog2_mask_cached(cache, index):
        mog2_warmup = mog2_warmup_frames_if_needed(
            cap, index, mog2_stream_frame_index
        )

    return previous, next_frame, mog2_warmup


def stereo_ball_mask_playback_inputs(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
    method: BallDetectionMethod,
    index: int,
    left_mog2_stream_frame_index: int | None,
    right_mog2_stream_frame_index: int | None,
    left_cache: object | None = None,
    right_cache: object | None = None,
    frame_count: int = 0,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
]:
    left_previous, left_next, left_mog2_warmup = ball_mask_playback_inputs(
        left_cap,
        method,
        index,
        left_mog2_stream_frame_index,
        left_cache,
        frame_count,
    )
    right_previous, right_next, right_mog2_warmup = ball_mask_playback_inputs(
        right_cap,
        method,
        index,
        right_mog2_stream_frame_index,
        right_cache,
        frame_count,
    )
    return (
        left_previous,
        right_previous,
        left_next,
        right_next,
        left_mog2_warmup,
        right_mog2_warmup,
    )


def filter_inputs_for_playback(
    cap: cv2.VideoCapture,
    frame_filter: FrameFilter,
    index: int,
    gru_stream_frame_index: int | None,
    mog2_stream_frame_index: int | None = None,
    cache: object | None = None,
    frame_count: int = 0,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    int | None,
]:
    previous: np.ndarray | None = None
    next_frame: np.ndarray | None = None
    mog2_warmup: list[np.ndarray] | None = None
    warmup_frames: list[np.ndarray] | None = None
    warmup_start_index: int | None = None

    if frame_filter.filter_id in BALL_MASK_FILTER_IDS:
        previous, next_frame, mog2_warmup = ball_mask_playback_inputs(
            cap,
            frame_filter.ball_detection_method,
            index,
            mog2_stream_frame_index,
            cache,
            frame_count,
        )

    if frame_filter.filter_id in (
        FilterId.GRU_THROW_INFERENCE,
        FilterId.TRAJECTORY_TRACKING,
        FilterId.STEREO_TRACKING,
    ):
        gru_cached = cache is not None and cache.has_gru(index)
        if not gru_cached:
            warmup_frames = gru_warmup_frames_if_needed(
                cap,
                index,
                gru_stream_frame_index,
                frame_filter.throw_buffer_size(),
            )
            if warmup_frames is not None:
                warmup_start_index = max(
                    0, index - frame_filter.throw_buffer_size() + 1
                )

    return previous, next_frame, mog2_warmup, warmup_frames, warmup_start_index


def apply_filter_to_frame(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    *,
    previous_frame: np.ndarray | None = None,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    warmup_start_index: int | None = None,
    video_fps: float | None = None,
    frame_index: int | None = None,
    cache: object | None = None,
) -> np.ndarray:
    return frame_filter.apply(
        frame,
        previous_frame=previous_frame,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        warmup_frames=warmup_frames,
        warmup_start_index=warmup_start_index,
        video_fps=video_fps,
        frame_index=frame_index,
        cache=cache,
    )


def frame_to_display_photo(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    display_size: tuple[int, int],
    *,
    previous_frame: np.ndarray | None = None,
    next_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    warmup_start_index: int | None = None,
    video_fps: float | None = None,
    frame_index: int | None = None,
    cache: object | None = None,
) -> ImageTk.PhotoImage:
    filtered = apply_filter_to_frame(
        frame_filter,
        frame,
        previous_frame=previous_frame,
        next_frame=next_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        warmup_frames=warmup_frames,
        warmup_start_index=warmup_start_index,
        video_fps=video_fps,
        frame_index=frame_index,
        cache=cache,
    )
    return frame_to_photo(filtered, display_size)


def gru_warmup_for_playback(
    cap: cv2.VideoCapture,
    index: int,
    gru_stream_frame_index: int | None,
    buffer_size: int,
    cache: object | None = None,
) -> tuple[list[np.ndarray] | None, int | None]:
    if cache is not None and cache.has_gru(index):
        return None, None
    warmup_frames = gru_warmup_frames_if_needed(
        cap, index, gru_stream_frame_index, buffer_size
    )
    if warmup_frames is None:
        return None, None
    return warmup_frames, max(0, index - buffer_size + 1)


def mog2_warmup_for_playback(
    cap: cv2.VideoCapture,
    index: int,
    mog2_stream_frame_index: int | None,
    cache: object | None = None,
    *,
    method: BallDetectionMethod = BallDetectionMethod.MOG2_CLOSING,
) -> list[np.ndarray] | None:
    if method == BallDetectionMethod.FRAME_DIFF:
        return None
    if method == BallDetectionMethod.MOG2_CLOSING:
        if cache is not None and cache.has_motion_mask(method, index):
            return None
    elif cache is not None and _mog2_mask_cached(cache, index):
        return None
    return mog2_warmup_frames_if_needed(cap, index, mog2_stream_frame_index)


def uses_gru_streaming(frame_filter: FrameFilter) -> bool:
    return frame_filter.filter_id in (
        FilterId.GRU_THROW_INFERENCE,
        FilterId.TRAJECTORY_TRACKING,
        FilterId.STEREO_TRACKING,
    )


def uses_mog2_streaming(frame_filter: FrameFilter) -> bool:
    return (
        uses_mog2_component(frame_filter.ball_detection_method)
        and frame_filter.filter_id in BALL_MASK_FILTER_IDS
    )
