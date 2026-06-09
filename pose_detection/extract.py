from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .detector import PoseDetector, detect_dominant_hand_detection
from .normalize import normalize_hand_keypoints
from .types import (
    DominantHand,
    DominantHandDetection,
    DominantHandSequence,
    NormalizedDominantHandSequence,
)


def _hand_to_arrays(hand: DominantHand | None) -> tuple[np.ndarray, int]:
    keypoints = np.full((3, 3), np.nan, dtype=np.float32)
    if hand is None:
        return keypoints, -1

    for joint_index, joint in enumerate(hand.joints):
        keypoints[joint_index, 0] = joint.x
        keypoints[joint_index, 1] = joint.y
        keypoints[joint_index, 2] = joint.confidence

    side_code = 0 if hand.side == "left" else 1
    return keypoints, side_code


def _detection_to_arrays(
    detection: DominantHandDetection | None,
) -> tuple[np.ndarray, int]:
    if detection is None:
        return _hand_to_arrays(None)
    return _hand_to_arrays(detection.hand)


def extract_dominant_hands(
    frames: Sequence[np.ndarray],
    detector: PoseDetector | None = None,
) -> DominantHandSequence:
    """Extract dominant-arm positions and confidences for every frame.

    Returns a DominantHandSequence whose keypoints array has shape
    (num_frames, 3, 3) — shoulder, elbow, wrist × (x, y, confidence). Frames
    with no valid dominant hand contain NaN positions and zero confidence.
    """
    pose_detector = detector or PoseDetector()
    num_frames = len(frames)
    keypoints = np.full((num_frames, 3, 3), np.nan, dtype=np.float32)
    sides = np.full(num_frames, -1, dtype=np.int8)

    for frame_index, frame in enumerate(frames):
        frame_keypoints, side_code = _detection_to_arrays(
            detect_dominant_hand_detection(frame, detector=pose_detector),
        )
        keypoints[frame_index] = frame_keypoints
        sides[frame_index] = side_code

    return DominantHandSequence(keypoints=keypoints, sides=sides)


def extract_normalized_dominant_hands(
    frames: Sequence[np.ndarray],
    detector: PoseDetector | None = None,
) -> NormalizedDominantHandSequence:
    """Extract torso-normalized dominant-arm keypoints for every frame.

    Normalized positions are relative to the dominant shoulder and scaled by
    the shoulder-to-hip length on the same side. Original image-space keypoints
    are included for debugging.
    """
    pose_detector = detector or PoseDetector()
    num_frames = len(frames)
    keypoints = np.full((num_frames, 3, 3), np.nan, dtype=np.float32)
    normalized_keypoints = np.full((num_frames, 3, 3), np.nan, dtype=np.float32)
    sides = np.full(num_frames, -1, dtype=np.int8)
    torso_scales = np.full(num_frames, np.nan, dtype=np.float32)
    anchors = np.full((num_frames, 2), np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(frames):
        detection = detect_dominant_hand_detection(frame, detector=pose_detector)
        frame_keypoints, side_code = _detection_to_arrays(detection)
        keypoints[frame_index] = frame_keypoints
        sides[frame_index] = side_code

        if detection is not None:
            normalized, scale, anchor = normalize_hand_keypoints(detection)
            normalized_keypoints[frame_index] = normalized
            torso_scales[frame_index] = scale
            anchors[frame_index, 0] = anchor[0]
            anchors[frame_index, 1] = anchor[1]

    return NormalizedDominantHandSequence(
        keypoints=keypoints,
        normalized_keypoints=normalized_keypoints,
        sides=sides,
        torso_scale=torso_scales,
        anchor=anchors,
    )
