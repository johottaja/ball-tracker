from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import cv2
import numpy as np

from .ball_motion import (
    BallDetectionMethod,
    HYBRID_METHODS,
    uses_frame_diff_component,
)
from .config import MOG2_HISTORY
from .playback import read_frame_at
from .stereo_timeline import StereoTimeline

Side = Literal["left", "right"]


class StereoFrameReader:
    """Read native source frames through a master stereo timeline."""

    def __init__(
        self,
        left_cap: cv2.VideoCapture,
        right_cap: cv2.VideoCapture,
        timeline: StereoTimeline,
    ) -> None:
        self._left_cap = left_cap
        self._right_cap = right_cap
        self._timeline = timeline
        self._left_pos = -1
        self._right_pos = -1
        self._left_frame: np.ndarray | None = None
        self._right_frame: np.ndarray | None = None
        self._frame_cache: dict[tuple[Side, int], np.ndarray] = {}

    @property
    def timeline(self) -> StereoTimeline:
        return self._timeline

    def release(self) -> None:
        self._left_cap.release()
        self._right_cap.release()

    def read_source(self, side: Side, source_index: int) -> np.ndarray | None:
        cache_key = (side, source_index)
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            return cached

        cap = self._left_cap if side == "left" else self._right_cap
        if side == "left":
            if source_index == self._left_pos and self._left_frame is not None:
                return self._left_frame
        elif source_index == self._right_pos and self._right_frame is not None:
            return self._right_frame

        ok, frame = read_frame_at(cap, source_index)
        if not ok or frame is None:
            return None

        if side == "left":
            self._left_pos = source_index
            self._left_frame = frame
        else:
            self._right_pos = source_index
            self._right_frame = frame
        self._frame_cache[cache_key] = frame
        return frame

    def read_at_master(
        self, master_index: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        left_frame = self.read_source(
            "left", self._timeline.left_source_index(master_index)
        )
        right_frame = self.read_source(
            "right", self._timeline.right_source_index(master_index)
        )
        return left_frame, right_frame

    def native_neighbor_frames(
        self, side: Side, master_index: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        previous_index, next_index = self._timeline.native_neighbor_indices(
            side, master_index
        )
        previous = (
            self.read_source(side, previous_index)
            if previous_index is not None
            else None
        )
        next_frame = (
            self.read_source(side, next_index) if next_index is not None else None
        )
        return previous, next_frame

    def mog2_warmup_frames(self, side: Side, master_index: int) -> list[np.ndarray]:
        source_index = self._timeline.source_index(side, master_index)
        if source_index <= 0:
            return []
        start = max(0, source_index - MOG2_HISTORY)
        frames: list[np.ndarray] = []
        for index in range(start, source_index):
            frame = self.read_source(side, index)
            if frame is not None:
                frames.append(frame)
        return frames

    def master_warmup_frames(
        self, side: Side, master_index: int, count: int
    ) -> tuple[list[np.ndarray], int | None]:
        if master_index <= 0 or count <= 0:
            return [], None
        start = max(0, master_index - count + 1)
        frames: list[np.ndarray] = []
        for index in range(start, master_index):
            frame = self.read_source(
                side, self._timeline.source_index(side, index)
            )
            if frame is not None:
                frames.append(frame)
        return frames, start


def gru_warmup_for_timeline_playback(
    reader: StereoFrameReader,
    master_index: int,
    gru_stream_frame_index: int | None,
    buffer_size: int,
    cache: object | None = None,
) -> tuple[list[np.ndarray] | None, int | None]:
    """Rebuild GRU history only after a seek; return None on sequential forward steps."""
    if cache is not None and cache.has_gru(master_index):
        return None, None
    if (
        gru_stream_frame_index is not None
        and master_index == gru_stream_frame_index + 1
    ):
        return None, None
    warmup_frames, warmup_start_index = reader.master_warmup_frames(
        "left", master_index, buffer_size - 1
    )
    if not warmup_frames:
        return None, None
    return warmup_frames, warmup_start_index


def tracker_throw_label_during_left_hold(
    throw_label: int,
    *,
    timeline: StereoTimeline | None,
    master_index: int | None,
    cache: object | None,
) -> int:
    """Keep the tracker throw label across left-camera holds.

    GRU is re-run on held pixels for the overlay; the logit often drops even though
    the throw is still in progress on the reference timeline. The trajectory tracker
    should not treat those duplicate slots as the throw ending.
    """
    if (
        timeline is None
        or master_index is None
        or master_index <= 0
        or not timeline.is_hold("left", master_index)
        or cache is None
        or not cache.has_gru(master_index - 1)
    ):
        return throw_label
    return cache.get_gru(master_index - 1).label


def iter_stereo_timeline(
    reader: StereoFrameReader,
    *,
    need_neighbors: bool,
) -> Iterator[
    tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]
]:
    timeline = reader.timeline
    for master_index in range(timeline.master_count):
        left_frame, right_frame = reader.read_at_master(master_index)
        if left_frame is None or right_frame is None:
            break

        if need_neighbors:
            left_previous, left_next = reader.native_neighbor_frames("left", master_index)
            right_previous, right_next = reader.native_neighbor_frames(
                "right", master_index
            )
        else:
            left_previous = left_next = right_previous = right_next = None

        yield (
            master_index,
            left_frame,
            right_frame,
            left_previous,
            right_previous,
            left_next,
            right_next,
        )


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


def stereo_timeline_ball_mask_inputs(
    reader: StereoFrameReader,
    method: BallDetectionMethod,
    master_index: int,
    left_mog2_stream_frame_index: int | None,
    right_mog2_stream_frame_index: int | None,
    left_cache: object | None = None,
    right_cache: object | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
]:
    left_previous = left_next = right_previous = right_next = None
    left_mog2_warmup = right_mog2_warmup = None

    if uses_frame_diff_component(method) and not _frame_diff_mask_cached(
        left_cache, master_index
    ):
        left_previous, left_next = reader.native_neighbor_frames("left", master_index)
        right_previous, right_next = reader.native_neighbor_frames("right", master_index)

    if method == BallDetectionMethod.MOG2_CLOSING:
        if left_cache is None or not left_cache.has_motion_mask(method, master_index):
            if left_mog2_stream_frame_index is None or master_index != (
                left_mog2_stream_frame_index + 1
            ):
                left_mog2_warmup = reader.mog2_warmup_frames("left", master_index)
        if right_cache is None or not right_cache.has_motion_mask(method, master_index):
            if right_mog2_stream_frame_index is None or master_index != (
                right_mog2_stream_frame_index + 1
            ):
                right_mog2_warmup = reader.mog2_warmup_frames("right", master_index)
    elif method in HYBRID_METHODS and (
        not _mog2_mask_cached(left_cache, master_index)
        or not _mog2_mask_cached(right_cache, master_index)
    ):
        if left_mog2_stream_frame_index is None or master_index != (
            left_mog2_stream_frame_index + 1
        ):
            left_mog2_warmup = reader.mog2_warmup_frames("left", master_index)
        if right_mog2_stream_frame_index is None or master_index != (
            right_mog2_stream_frame_index + 1
        ):
            right_mog2_warmup = reader.mog2_warmup_frames("right", master_index)

    return (
        left_previous,
        right_previous,
        left_next,
        right_next,
        left_mog2_warmup,
        right_mog2_warmup,
    )


def stereo_timeline_filter_inputs(
    reader: StereoFrameReader,
    side: Side,
    master_index: int,
    *,
    ball_method: BallDetectionMethod | None,
    uses_ball_mask: bool,
    uses_gru: bool,
    gru_stream_frame_index: int | None,
    mog2_stream_frame_index: int | None,
    gru_buffer_size: int,
    cache: object | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    list[np.ndarray] | None,
    list[np.ndarray] | None,
    int | None,
]:
    previous = next_frame = None
    mog2_warmup = None
    warmup_frames = None
    warmup_start_index = None

    if uses_ball_mask and ball_method is not None:
        if uses_frame_diff_component(ball_method) and not _frame_diff_mask_cached(
            cache, master_index
        ):
            previous, next_frame = reader.native_neighbor_frames(side, master_index)

        if ball_method == BallDetectionMethod.MOG2_CLOSING:
            if cache is None or not cache.has_motion_mask(ball_method, master_index):
                if mog2_stream_frame_index is None or master_index != (
                    mog2_stream_frame_index + 1
                ):
                    mog2_warmup = reader.mog2_warmup_frames(side, master_index)
        elif ball_method in HYBRID_METHODS and not _mog2_mask_cached(
            cache, master_index
        ):
            if mog2_stream_frame_index is None or master_index != (
                mog2_stream_frame_index + 1
            ):
                mog2_warmup = reader.mog2_warmup_frames(side, master_index)

    if uses_gru and (cache is None or not cache.has_gru(master_index)):
        warmup_frames, warmup_start_index = gru_warmup_for_timeline_playback(
            reader,
            master_index,
            gru_stream_frame_index,
            gru_buffer_size,
            cache,
        )

    return previous, next_frame, mog2_warmup, warmup_frames, warmup_start_index
