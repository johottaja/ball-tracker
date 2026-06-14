from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pose_detection import (
    DominantHandDetection,
    PoseDetector,
    detect_dominant_hand_detection,
    normalize_hand_keypoints,
)

from .config import MODELS_DIR
from .model import load_throw_model


@dataclass(frozen=True)
class ThrowPrediction:
    label: int
    logit: float
    probability: float
    has_pose: bool
    detection: DominantHandDetection | None


def list_throw_models() -> list[Path]:
    if not MODELS_DIR.is_dir():
        return []
    return sorted(MODELS_DIR.glob("*.pt"), key=lambda path: path.name.lower())


def default_throw_model_path() -> Path | None:
    if not MODELS_DIR.is_dir():
        return None
    models = sorted(
        MODELS_DIR.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return models[0] if models else None


def features_from_detection(detection: DominantHandDetection | None) -> np.ndarray:
    """Elbow and wrist normalized x,y — shape (4,), NaN when pose is missing."""
    features = np.full(4, np.nan, dtype=np.float32)
    if detection is None:
        return features

    normalized, _, _ = normalize_hand_keypoints(detection)
    features[0:2] = normalized[1, :2]
    features[2:4] = normalized[2, :2]
    return features


def features_from_frame(
    frame: np.ndarray,
    *,
    detector: PoseDetector | None = None,
) -> tuple[np.ndarray, DominantHandDetection | None]:
    detection = detect_dominant_hand_detection(frame, detector=detector)
    return features_from_detection(detection), detection


def _build_window(features_history: Sequence[np.ndarray], buffer_size: int) -> np.ndarray:
    """Causal rolling window — shape (1, buffer_size, 4), early NaNs zeroed."""
    history = list(features_history)
    window = np.full((buffer_size, 4), np.nan, dtype=np.float32)
    pad_count = buffer_size - len(history)
    if pad_count > 0:
        window[pad_count:] = np.stack(history)
    else:
        window[:] = np.stack(history[-buffer_size:])
    return np.nan_to_num(window[np.newaxis], nan=0.0)


class ThrowInference:
    """Streaming GRU throw classifier for single-frame inference."""

    def __init__(
        self,
        model_path: Path,
        *,
        detector: PoseDetector | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.model_path = model_path
        self.model, metadata = load_throw_model(model_path, map_location=map_location)
        self.buffer_size = int(metadata["buffer_size"])
        self.metadata = metadata
        self.detector = detector or PoseDetector()
        self._feature_history: deque[np.ndarray] = deque(maxlen=self.buffer_size)

    def reset(self) -> None:
        self._feature_history.clear()

    def _push_frame(self, frame: np.ndarray) -> tuple[np.ndarray, DominantHandDetection | None]:
        features, detection = features_from_frame(frame, detector=self.detector)
        self._feature_history.append(features)
        return features, detection

    def predict(
        self,
        frame: np.ndarray,
        *,
        warmup_frames: Sequence[np.ndarray] | None = None,
    ) -> ThrowPrediction:
        """Classify one frame. Optional warmup_frames rebuild history after a seek."""
        if warmup_frames is not None:
            self.reset()
            for warmup_frame in warmup_frames:
                self._push_frame(warmup_frame)

        _, detection = self._push_frame(frame)
        has_pose = detection is not None
        if not has_pose:
            return ThrowPrediction(
                label=0,
                logit=0.0,
                probability=0.0,
                has_pose=False,
                detection=None,
            )

        window = _build_window(self._feature_history, self.buffer_size)
        with torch.no_grad():
            logit = self.model(torch.from_numpy(window)).item()

        probability = float(torch.sigmoid(torch.tensor(logit)).item())
        label = 1 if probability >= 0.75 else 0
        return ThrowPrediction(
            label=label,
            logit=logit,
            probability=probability,
            has_pose=True,
            detection=detection,
        )
