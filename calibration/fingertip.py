from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INDEX_FINGERTIP_LANDMARK = 8
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class FingertipDetection:
    x: float
    y: float


def ensure_hand_landmarker_model(model_path: Path) -> Path:
    """Download MediaPipe's official hand-landmarker bundle on first use."""
    if model_path.is_file():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        urlretrieve(HAND_LANDMARKER_MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except (OSError, URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Could not download the MediaPipe hand model to {model_path}: {exc}"
        ) from exc
    return model_path


class FingertipDetector:
    """MediaPipe index-fingertip detector for calibration feeds."""

    def __init__(self, model_path: Path) -> None:
        model_path = ensure_hand_landmarker_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def detect_nearest(
        self,
        frame_bgr: np.ndarray,
        expected_xy: tuple[float, float],
        *,
        max_distance_px: float,
    ) -> FingertipDetection | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(image)
        height, width = frame_bgr.shape[:2]
        expected = np.array(expected_xy, dtype=np.float64)
        candidates: list[tuple[float, FingertipDetection]] = []
        for landmarks in result.hand_landmarks:
            tip = landmarks[INDEX_FINGERTIP_LANDMARK]
            point = FingertipDetection(x=tip.x * width, y=tip.y * height)
            distance = float(np.linalg.norm(np.array([point.x, point.y]) - expected))
            candidates.append((distance, point))
        if not candidates:
            return None
        distance, point = min(candidates, key=lambda candidate: candidate[0])
        return point if distance <= max_distance_px else None
