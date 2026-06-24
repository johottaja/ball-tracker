from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageTk

from .config import FRAME_WINDOW_SIZE
from .display import frame_to_photo
from .filters import PREV_FRAME_DIFF_FILTER_IDS, FilterId, FrameFilter


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


def window_frames_for_diff(
    cap: cv2.VideoCapture, index: int, window_size: int = FRAME_WINDOW_SIZE
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    start = max(0, index - window_size)
    for frame_index in range(start, index):
        ok, frame = read_frame_at(cap, frame_index)
        if ok and frame is not None:
            frames.append(frame)
    return frames


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
) -> tuple[np.ndarray | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    previous: np.ndarray | None = None
    window_frames: list[np.ndarray] | None = None
    warmup_frames: list[np.ndarray] | None = None

    if frame_filter.filter_id in PREV_FRAME_DIFF_FILTER_IDS:
        previous = previous_frame_for_diff(cap, index)
    elif frame_filter.filter_id == FilterId.FRAME_DIFF_WINDOW:
        window_frames = window_frames_for_diff(cap, index, frame_filter.window_size)
    elif frame_filter.filter_id in (
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

    return previous, window_frames, warmup_frames


def apply_filter_to_frame(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    *,
    previous_frame: np.ndarray | None = None,
    window_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    video_fps: float | None = None,
) -> np.ndarray:
    return frame_filter.apply(
        frame,
        previous_frame=previous_frame,
        window_frames=window_frames,
        warmup_frames=warmup_frames,
        video_fps=video_fps,
    )


def frame_to_display_photo(
    frame_filter: FrameFilter,
    frame: np.ndarray,
    display_size: tuple[int, int],
    *,
    previous_frame: np.ndarray | None = None,
    window_frames: list[np.ndarray] | None = None,
    warmup_frames: list[np.ndarray] | None = None,
    video_fps: float | None = None,
) -> ImageTk.PhotoImage:
    filtered = apply_filter_to_frame(
        frame_filter,
        frame,
        previous_frame=previous_frame,
        window_frames=window_frames,
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
