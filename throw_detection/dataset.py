from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pose_detection.types import NormalizedDominantHandSequence
from training_recorder.paths import sanitize_training_set_name

from .config import BUFFER_SIZE, REPO_ROOT, TRAINING_SETS_DIR
from .features import frame_features_from_sequence, rolling_windows


def dataset_path_for_set(set_name: str) -> Path:
    sanitized = sanitize_training_set_name(set_name)
    return TRAINING_SETS_DIR / f"{sanitized}.npz"


def _clip_path_to_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _clip_path_from_stored(stored: str) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass
class LabelingSession:
    set_name: str
    clip_paths: list[Path]
    labels: dict[Path, np.ndarray] = field(default_factory=dict)
    pose_cache: dict[Path, NormalizedDominantHandSequence] = field(
        default_factory=dict,
    )

    @property
    def sanitized_set_name(self) -> str:
        return sanitize_training_set_name(self.set_name)

    def label_array_for_clip(self, clip_path: Path, frame_count: int) -> np.ndarray:
        resolved = clip_path.resolve()
        existing = self.labels.get(resolved)
        if existing is not None:
            if len(existing) == frame_count:
                return existing
            resized = np.zeros(frame_count, dtype=np.int8)
            copy_len = min(len(existing), frame_count)
            resized[:copy_len] = existing[:copy_len]
            self.labels[resolved] = resized
            return resized
        labels = np.zeros(frame_count, dtype=np.int8)
        self.labels[resolved] = labels
        return labels


def _build_concatenated_arrays(
    session: LabelingSession,
    buffer_size: int,
) -> dict[str, np.ndarray]:
    clip_paths = session.clip_paths
    clip_frame_counts: list[int] = []
    clip_offsets: list[int] = []
    labels_parts: list[np.ndarray] = []
    features_parts: list[np.ndarray] = []
    windows_parts: list[np.ndarray] = []
    sides_parts: list[np.ndarray] = []
    offset = 0

    for clip_path in clip_paths:
        resolved = clip_path.resolve()
        sequence = session.pose_cache.get(resolved)
        if sequence is None:
            raise ValueError(f"Missing pose cache for clip: {clip_path}")

        frame_count = len(sequence.sides)
        clip_frame_counts.append(frame_count)
        clip_offsets.append(offset)

        labels = session.label_array_for_clip(resolved, frame_count)
        features = frame_features_from_sequence(sequence)
        windows = rolling_windows(features, buffer_size)

        labels_parts.append(labels)
        features_parts.append(features)
        windows_parts.append(windows)
        sides_parts.append(sequence.sides.astype(np.int8))
        offset += frame_count

    return {
        "set_name": np.array(session.sanitized_set_name),
        "buffer_size": np.array(buffer_size),
        "clip_paths": np.array(
            [_clip_path_to_relative(path) for path in clip_paths],
            dtype=object,
        ),
        "clip_frame_counts": np.array(clip_frame_counts, dtype=np.int32),
        "clip_offsets": np.array(clip_offsets, dtype=np.int32),
        "labels": np.concatenate(labels_parts) if labels_parts else np.array([], dtype=np.int8),
        "frame_features": (
            np.concatenate(features_parts)
            if features_parts
            else np.zeros((0, 4), dtype=np.float32)
        ),
        "windows": (
            np.concatenate(windows_parts)
            if windows_parts
            else np.zeros((0, buffer_size, 4), dtype=np.float32)
        ),
        "sides": (
            np.concatenate(sides_parts)
            if sides_parts
            else np.array([], dtype=np.int8)
        ),
    }


def save_dataset(path: Path, session: LabelingSession) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _build_concatenated_arrays(session, BUFFER_SIZE)
    np.savez_compressed(path, **arrays)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def merge_labels_from_file(session: LabelingSession, path: Path) -> bool:
    if not path.is_file():
        return False

    data = load_dataset(path)
    stored_paths = [_clip_path_from_stored(str(value)) for value in data["clip_paths"]]
    stored_labels = data["labels"]

    if "clip_offsets" in data and "clip_frame_counts" in data:
        offsets = data["clip_offsets"]
        counts = data["clip_frame_counts"]
        for clip_path, offset, count in zip(stored_paths, offsets, counts, strict=True):
            resolved = clip_path.resolve()
            if resolved not in {p.resolve() for p in session.clip_paths}:
                continue
            end = int(offset) + int(count)
            session.labels[resolved] = stored_labels[offset:end].astype(np.int8).copy()
        return True

    # Fallback: match clips in order
    offset = 0
    for clip_path, stored_path in zip(session.clip_paths, stored_paths, strict=False):
        resolved = clip_path.resolve()
        if resolved != stored_path.resolve():
            continue
        sequence = session.pose_cache.get(resolved)
        if sequence is None:
            continue
        count = len(sequence.sides)
        session.labels[resolved] = stored_labels[offset : offset + count].astype(
            np.int8,
        ).copy()
        offset += count
    return True
