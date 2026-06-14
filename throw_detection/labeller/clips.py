from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from pose_detection import PoseDetector
from pose_detection.detector import detect_dominant_hand_detection
from pose_detection.extract import _detection_to_arrays
from pose_detection.normalize import normalize_hand_keypoints
from pose_detection.types import NormalizedDominantHandSequence
from training_recorder.paths import training_set_dir


def list_clips(set_name: str) -> list[Path]:
    directory = training_set_dir(set_name)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("clip_*.mp4"))


def open_clip(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    return cap


def clip_frame_count(cap: cv2.VideoCapture) -> int:
    return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def read_frame_at(cap: cv2.VideoCapture, index: int) -> tuple[bool, np.ndarray | None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame


def extract_pose_from_video(
    path: Path,
    detector: PoseDetector,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> NormalizedDominantHandSequence:
    """Run pose detection frame-by-frame from disk without loading the whole clip."""
    cap = open_clip(path)
    estimated_total = clip_frame_count(cap)

    keypoint_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    sides: list[int] = []
    torso_scales: list[float] = []
    anchor_rows: list[np.ndarray] = []

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detection = detect_dominant_hand_detection(frame, detector=detector)
            frame_keypoints, side_code = _detection_to_arrays(detection)
            normalized = np.full((3, 3), np.nan, dtype=np.float32)
            scale = np.nan
            anchor = np.array([np.nan, np.nan], dtype=np.float32)

            if detection is not None:
                normalized, scale, anchor_xy = normalize_hand_keypoints(detection)
                anchor[0] = anchor_xy[0]
                anchor[1] = anchor_xy[1]

            keypoint_rows.append(frame_keypoints)
            normalized_rows.append(normalized)
            sides.append(side_code)
            torso_scales.append(scale)
            anchor_rows.append(anchor)

            frame_index += 1
            if progress is not None:
                total = estimated_total if estimated_total > 0 else frame_index
                progress(frame_index, total)
    finally:
        cap.release()

    if not keypoint_rows:
        return NormalizedDominantHandSequence(
            keypoints=np.zeros((0, 3, 3), dtype=np.float32),
            normalized_keypoints=np.zeros((0, 3, 3), dtype=np.float32),
            sides=np.zeros(0, dtype=np.int8),
            torso_scale=np.zeros(0, dtype=np.float32),
            anchor=np.zeros((0, 2), dtype=np.float32),
        )

    return NormalizedDominantHandSequence(
        keypoints=np.stack(keypoint_rows),
        normalized_keypoints=np.stack(normalized_rows),
        sides=np.array(sides, dtype=np.int8),
        torso_scale=np.array(torso_scales, dtype=np.float32),
        anchor=np.stack(anchor_rows),
    )
