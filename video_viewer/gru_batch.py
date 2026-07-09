from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from throw_detection.features import rolling_windows
from throw_detection.inference import ThrowPrediction, features_from_detection
from throw_detection.model import load_throw_model
from video_viewer.playback_cache import PlaybackCache, StreamPlaybackCache
from video_viewer.yolo_batch import MonoYoloInferenceStore, StereoYoloInferenceStore

from .config import GRU_BATCH_SIZE

GruCacheStatus = Literal["missing", "ready", "stale", "wrong_model", "no_model"]


@dataclass(frozen=True)
class GruBatchProgress:
    frame_index: int
    frame_count: int
    elapsed_s: float

    @property
    def fraction(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        return (self.frame_index + 1) / self.frame_count

    @property
    def eta_s(self) -> float | None:
        done = self.frame_index + 1
        if done <= 0 or self.frame_count <= done:
            return None
        rate = self.elapsed_s / done
        remaining = self.frame_count - done
        return rate * remaining


GruProgressCallback = Callable[[GruBatchProgress], None]

_THROW_LABEL_THRESHOLD = 0.75


@dataclass(frozen=True)
class GruInferenceStore:
    frame_count: int
    model_path: str
    labels: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    has_pose: np.ndarray

    def prediction(
        self,
        frame_index: int,
        *,
        detection: object | None,
    ) -> ThrowPrediction:
        from pose_detection import DominantHandDetection

        det = detection if isinstance(detection, DominantHandDetection) else None
        return ThrowPrediction(
            label=int(self.labels[frame_index]),
            logit=float(self.logits[frame_index]),
            probability=float(self.probabilities[frame_index]),
            has_pose=bool(self.has_pose[frame_index]),
            detection=det if self.has_pose[frame_index] else None,
        )


def _resolved_model_path(model_path: Path) -> Path:
    return model_path.resolve()


def _model_cache_key(model_path: Path) -> str:
    return str(_resolved_model_path(model_path))


def gru_cache_status(
    path: Path,
    expected_frame_count: int,
    model_path: Path | None,
) -> GruCacheStatus:
    if model_path is None or not model_path.is_file():
        return "no_model"
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as data:
            frame_count = int(data["frame_count"])
            cached_model = str(data["model_path"])
    except (OSError, KeyError, TypeError, ValueError):
        return "missing"
    if cached_model != _model_cache_key(model_path):
        return "wrong_model"
    if expected_frame_count > 0 and frame_count != expected_frame_count:
        return "stale"
    return "ready"


def gru_cache_status_label(
    status: GruCacheStatus,
    *,
    cached_frames: int = 0,
    expected_frames: int = 0,
) -> str:
    if status == "no_model":
        return "GRU cache: no model"
    if status == "missing":
        return "GRU cache: not run"
    if status == "wrong_model":
        return "GRU cache: wrong model — re-run"
    if status == "stale":
        return (
            f"GRU cache: stale (cached {cached_frames:,}, "
            f"video {expected_frames:,}) — re-run"
        )
    return f"GRU cache: ready ({cached_frames:,} frames)"


def save_gru_inferences(path: Path, store: GruInferenceStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(
        tmp,
        frame_count=store.frame_count,
        model_path=store.model_path,
        labels=store.labels,
        logits=store.logits,
        probabilities=store.probabilities,
        has_pose=store.has_pose,
    )
    tmp.replace(path)


def load_gru_inferences(path: Path) -> GruInferenceStore:
    with np.load(path, allow_pickle=False) as data:
        return GruInferenceStore(
            frame_count=int(data["frame_count"]),
            model_path=str(data["model_path"]),
            labels=data["labels"],
            logits=data["logits"],
            probabilities=data["probabilities"],
            has_pose=data["has_pose"],
        )


def _features_from_mono_yolo(store: MonoYoloInferenceStore) -> np.ndarray:
    features = np.full((store.frame_count, 4), np.nan, dtype=np.float32)
    for frame_index in range(store.frame_count):
        detection = store.detection(frame_index)
        if detection is not None:
            features[frame_index] = features_from_detection(detection)
    return features


def _features_from_stereo_yolo(store: StereoYoloInferenceStore) -> np.ndarray:
    features = np.full((store.frame_count, 4), np.nan, dtype=np.float32)
    for frame_index in range(store.frame_count):
        detection = store.detection("left", frame_index)
        if detection is not None:
            features[frame_index] = features_from_detection(detection)
    return features


def _run_gru_on_features(
    features: np.ndarray,
    *,
    model_path: Path,
    batch_size: int = GRU_BATCH_SIZE,
    progress: GruProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GruInferenceStore:
    frame_count = len(features)
    has_pose = ~np.isnan(features).all(axis=1)
    labels = np.zeros(frame_count, dtype=np.int8)
    logits = np.zeros(frame_count, dtype=np.float32)
    probabilities = np.zeros(frame_count, dtype=np.float32)

    if not has_pose.any():
        return GruInferenceStore(
            frame_count=frame_count,
            model_path=_model_cache_key(model_path),
            labels=labels,
            logits=logits,
            probabilities=probabilities,
            has_pose=has_pose,
        )

    model, metadata = load_throw_model(model_path)
    buffer_size = int(metadata["buffer_size"])
    windows = np.nan_to_num(rolling_windows(features, buffer_size), nan=0.0)

    pose_indices = np.flatnonzero(has_pose)
    start = time.monotonic()

    for batch_start in range(0, len(pose_indices), batch_size):
        if cancel_check is not None and cancel_check():
            break

        batch_indices = pose_indices[batch_start : batch_start + batch_size]
        batch_windows = windows[batch_indices]
        with torch.no_grad():
            batch_logits = model(torch.from_numpy(batch_windows)).numpy()

        batch_probs = 1.0 / (1.0 + np.exp(-batch_logits))
        for offset, frame_index in enumerate(batch_indices):
            logit = float(batch_logits[offset])
            probability = float(batch_probs[offset])
            logits[frame_index] = logit
            probabilities[frame_index] = probability
            labels[frame_index] = 1 if probability >= _THROW_LABEL_THRESHOLD else 0

        if progress is not None:
            last_index = int(batch_indices[-1])
            progress(
                GruBatchProgress(
                    frame_index=last_index,
                    frame_count=frame_count,
                    elapsed_s=time.monotonic() - start,
                )
            )

    return GruInferenceStore(
        frame_count=frame_count,
        model_path=_model_cache_key(model_path),
        labels=labels,
        logits=logits,
        probabilities=probabilities,
        has_pose=has_pose,
    )


def run_mono_gru_inference_phase(
    *,
    yolo_store: MonoYoloInferenceStore,
    model_path: Path,
    output_path: Path,
    batch_size: int = GRU_BATCH_SIZE,
    progress: GruProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GruInferenceStore:
    features = _features_from_mono_yolo(yolo_store)
    store = _run_gru_on_features(
        features,
        model_path=model_path,
        batch_size=batch_size,
        progress=progress,
        cancel_check=cancel_check,
    )
    save_gru_inferences(output_path, store)
    return store


def run_stereo_gru_inference_phase(
    *,
    yolo_store: StereoYoloInferenceStore,
    model_path: Path,
    output_path: Path,
    batch_size: int = GRU_BATCH_SIZE,
    progress: GruProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GruInferenceStore:
    features = _features_from_stereo_yolo(yolo_store)
    store = _run_gru_on_features(
        features,
        model_path=model_path,
        batch_size=batch_size,
        progress=progress,
        cancel_check=cancel_check,
    )
    save_gru_inferences(output_path, store)
    return store


def populate_mono_gru_cache(
    gru_store: GruInferenceStore,
    cache: StreamPlaybackCache,
) -> None:
    for frame_index in range(gru_store.frame_count):
        detection = cache.get_pose(frame_index) if cache.has_pose(frame_index) else None
        cache.put_gru(
            frame_index,
            gru_store.prediction(frame_index, detection=detection),
        )


def populate_stereo_gru_cache(
    gru_store: GruInferenceStore,
    cache: PlaybackCache,
) -> None:
    populate_mono_gru_cache(gru_store, cache.main)


def populate_mono_gru_cache_with_yolo(
    gru_store: GruInferenceStore,
    yolo_store: MonoYoloInferenceStore,
    cache: StreamPlaybackCache,
) -> None:
    for frame_index in range(gru_store.frame_count):
        cache.put_gru(
            frame_index,
            gru_store.prediction(
                frame_index,
                detection=yolo_store.detection(frame_index),
            ),
        )


def populate_stereo_gru_cache_with_yolo(
    gru_store: GruInferenceStore,
    yolo_store: StereoYoloInferenceStore,
    cache: PlaybackCache,
) -> None:
    for frame_index in range(gru_store.frame_count):
        cache.main.put_gru(
            frame_index,
            gru_store.prediction(
                frame_index,
                detection=yolo_store.detection("left", frame_index),
            ),
        )


def try_load_gru_cache(
    path: Path,
    expected_frame_count: int,
    model_path: Path | None,
    cache: PlaybackCache,
    *,
    layout: Literal["mono", "stereo"],
) -> bool:
    if gru_cache_status(path, expected_frame_count, model_path) != "ready":
        return False
    gru_store = load_gru_inferences(path)
    if layout == "mono":
        populate_mono_gru_cache(gru_store, cache.main)
    else:
        populate_stereo_gru_cache(gru_store, cache)
    return True
