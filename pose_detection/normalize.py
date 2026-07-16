from __future__ import annotations

import numpy as np

from .detector import torso_scale
from .types import DominantHandDetection


def normalize_hand_keypoints(
    detection: DominantHandDetection,
    *,
    mirror_x: bool = False,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Convert arm keypoints to torso-normalized coordinates.

    Positions are offset by the dominant shoulder and divided by the
    shoulder-to-hip length on the same side. ``mirror_x`` canonicalizes
    horizontal motion for an opposite-side player. Confidences are unchanged.
    """
    hand = detection.hand
    normalized = np.full((3, 3), np.nan, dtype=np.float32)
    shoulder = hand.joints[0]
    anchor = (shoulder.x, shoulder.y)

    scale = torso_scale(detection.person_keypoints, hand.side)
    if scale is None:
        return normalized, np.nan, anchor

    for joint_index, joint in enumerate(hand.joints):
        normalized_x = (joint.x - anchor[0]) / scale
        normalized[joint_index, 0] = -normalized_x if mirror_x else normalized_x
        normalized[joint_index, 1] = (joint.y - anchor[1]) / scale
        normalized[joint_index, 2] = joint.confidence

    return normalized, scale, anchor
