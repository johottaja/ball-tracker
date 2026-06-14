from __future__ import annotations

import numpy as np

from pose_detection.types import NormalizedDominantHandSequence


def frame_features_from_sequence(
    sequence: NormalizedDominantHandSequence,
) -> np.ndarray:
    """Per-frame elbow and wrist normalized x,y — shape (num_frames, 4)."""
    elbow = sequence.normalized_keypoints[:, 1, :2]
    wrist = sequence.normalized_keypoints[:, 2, :2]
    return np.concatenate([elbow, wrist], axis=1).astype(np.float32)


def rolling_windows(
    features: np.ndarray,
    buffer_size: int,
) -> np.ndarray:
    """Causal rolling windows ending at each frame — shape (num_frames, buffer_size, 4).

    Frames before enough history are left-padded with NaN.
    """
    num_frames, feature_dim = features.shape
    windows = np.full(
        (num_frames, buffer_size, feature_dim),
        np.nan,
        dtype=np.float32,
    )
    for frame_index in range(num_frames):
        start = max(0, frame_index - buffer_size + 1)
        window_slice = features[start : frame_index + 1]
        pad_count = buffer_size - len(window_slice)
        if pad_count > 0:
            windows[frame_index, pad_count:] = window_slice
        else:
            windows[frame_index] = window_slice
    return windows
