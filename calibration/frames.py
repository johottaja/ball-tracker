from __future__ import annotations

import cv2
import numpy as np

from video_viewer.playback import read_frame_at


def capture_stereo_pair(
    *,
    mode: str,
    frame_index: int,
    left_last_raw: np.ndarray | None,
    right_last_raw: np.ndarray | None,
    left_cap: cv2.VideoCapture | None,
    right_cap: cv2.VideoCapture | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return copies of the current left/right frames for calibration, or None if unavailable."""
    if mode == "record":
        if left_last_raw is None or right_last_raw is None:
            return None
        return left_last_raw.copy(), right_last_raw.copy()

    if left_cap is None or right_cap is None or not left_cap.isOpened() or not right_cap.isOpened():
        return None

    ok_left, left_frame = read_frame_at(left_cap, frame_index)
    ok_right, right_frame = read_frame_at(right_cap, frame_index)
    if not ok_left or left_frame is None or not ok_right or right_frame is None:
        return None
    return left_frame.copy(), right_frame.copy()
