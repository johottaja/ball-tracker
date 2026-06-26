from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageTk

from .ball_motion import BallDetectionMethod
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


def filter_inputs_for_playback(
    cap: cv2.VideoCapture,
    frame_filter: FrameFilter,
    index: int,
    gru_stream_frame_index: int | None,
    mog2_stream_frame_index: int | None = None,
) -> tuple[np.ndarray | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    previous: np.ndarray | None = None
    mog2_warmup: list[np.ndarray] | None = None
    warmup_frames: list[np.ndarray] | None = None

    if frame_filter.filter_id in BALL_MASK_FILTER_IDS:
        if frame_filter.ball_detection_method == BallDetectionMethod.FRAME_DIFF:
            previous = previous_frame_for_diff(cap, index)
        else:
            mog2_warmup = mog2_warmup_frames_if_needed(
                cap, index, mog2_stream_frame_index
            )

    if frame_filter.filter_id in (
        FilterId.GRU_THROW_INFERENCE,
        FilterId.TRAJECTORY_TRACKING,
        FilterId.STEREO_TRACKING,
    ):
        warmup_frames = gru_warmup_frames_if_needed(
            cap,
            index,
            gru_stream_frame_index,
            frame_filter.throw_buffer_size(),
        )

    return previous, mog2_warmup, warmup_frames


def apply_filter_to_frame(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    *,
    previous_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    video_fps: float | None = None,
) -> np.ndarray:
    return frame_filter.apply(
        frame,
        previous_frame=previous_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        warmup_frames=warmup_frames,
        video_fps=video_fps,
    )


def frame_to_display_photo(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    display_size: tuple[int, int],
    *,
    previous_frame: np.ndarray | None = None,
    mog2_warmup_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    video_fps: float | None = None,
) -> ImageTk.PhotoImage:
    filtered = apply_filter_to_frame(
        frame_filter,
        frame,
        previous_frame=previous_frame,
        mog2_warmup_frames=mog2_warmup_frames,
        warmup_frames=warmup_frames,
        video_fps=video_fps,
    )
    return frame_to_photo(filtered, display_size)


def uses_gru_streaming(frame_filter: FrameFilter) -> bool:
    return frame_filter.filter_id in (
        FilterId.GRU_THROW_INFERENCE,
        FilterId.TRAJECTORY_TRACKING,
        FilterId.STEREO_TRACKING,
    )


def uses_mog2_streaming(frame_filter: FrameFilter) -> bool:
    return (
        frame_filter.ball_detection_method == BallDetectionMethod.MOG2_CLOSING
        and frame_filter.filter_id in BALL_MASK_FILTER_IDS
    )
