from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class Joint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class DominantHand:
    side: HandSide
    joints: tuple[Joint, ...]


@dataclass(frozen=True)
class DominantHandDetection:
    hand: DominantHand
    person_keypoints: np.ndarray
    """Full COCO keypoints for the selected person, shape (17, 3)."""


@dataclass(frozen=True)
class DominantHandSequence:
    """Dominant-arm keypoints for a sequence of frames."""

    keypoints: np.ndarray
    """Shape (num_frames, 3, 3): joint index × (x, y, confidence). NaN when missing."""

    sides: np.ndarray
    """Shape (num_frames,): -1 if undetected, 0 for left, 1 for right."""


@dataclass(frozen=True)
class NormalizedDominantHandSequence:
    """Dominant-arm keypoints in image and torso-normalized coordinates."""

    keypoints: np.ndarray
    """Shape (num_frames, 3, 3): original x, y, confidence. NaN when missing."""

    normalized_keypoints: np.ndarray
    """Shape (num_frames, 3, 3): shoulder-relative, torso-scaled x, y, confidence."""

    sides: np.ndarray
    """Shape (num_frames,): -1 if undetected, 0 for left, 1 for right."""

    torso_scale: np.ndarray
    """Shape (num_frames,): dominant shoulder-to-hip length used as scale. NaN when missing."""

    anchor: np.ndarray
    """Shape (num_frames, 2): dominant shoulder (x, y) used as offset. NaN when missing."""
