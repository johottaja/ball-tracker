from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from pose_detection import (
    DominantHandDetection,
    PoseDetector,
    dominant_hand_detection_from_keypoints,
    select_dominant_hand_detection,
)
from pose_detection.types import HandSide
from video_viewer.config import YOLO_BATCH_SIZE
from video_viewer.playback_cache import PlaybackCache, StreamPlaybackCache
from video_viewer.stereo_playback import StereoFrameReader
from video_viewer.stereo_timeline import load_stereo_timeline_for_videos

YoloCacheLayout = Literal["mono", "stereo"]
YoloCacheStatus = Literal["missing", "ready", "stale", "wrong_layout"]


@dataclass(frozen=True)
class YoloBatchProgress:
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


YoloProgressCallback = Callable[[YoloBatchProgress], None]


def _side_to_int(side: HandSide | None) -> int:
    if side is None:
        return -1
    return 0 if side == "left" else 1


def _int_to_side(value: int) -> HandSide | None:
    if value < 0:
        return None
    return "left" if value == 0 else "right"


@dataclass(frozen=True)
class MonoYoloInferenceStore:
    frame_count: int
    person_keypoints: np.ndarray
    sides: np.ndarray

    def detection(self, frame_index: int) -> DominantHandDetection | None:
        person_keypoints = self.person_keypoints[frame_index]
        side = _int_to_side(int(self.sides[frame_index]))
        if side is None or np.isnan(person_keypoints).all():
            return None
        return dominant_hand_detection_from_keypoints(person_keypoints, side)


@dataclass(frozen=True)
class StereoYoloInferenceStore:
    frame_count: int
    left_person_keypoints: np.ndarray
    left_sides: np.ndarray
    right_person_keypoints: np.ndarray
    right_sides: np.ndarray

    def detection(self, camera: str, frame_index: int) -> DominantHandDetection | None:
        if camera == "left":
            person_keypoints = self.left_person_keypoints[frame_index]
            side = _int_to_side(int(self.left_sides[frame_index]))
        else:
            person_keypoints = self.right_person_keypoints[frame_index]
            side = _int_to_side(int(self.right_sides[frame_index]))
        if side is None or np.isnan(person_keypoints).all():
            return None
        return dominant_hand_detection_from_keypoints(person_keypoints, side)


# Backward-compatible alias for game_tracker.
YoloInferenceStore = StereoYoloInferenceStore


def _empty_person_array(count: int) -> np.ndarray:
    return np.full((count, 17, 3), np.nan, dtype=np.float32)


def _empty_sides_array(count: int) -> np.ndarray:
    return np.full(count, -1, dtype=np.int8)


def _layout_from_npz(data: np.lib.npyio.NpzFile) -> YoloCacheLayout | None:
    if "layout" in data:
        layout = str(data["layout"])
        if layout in ("mono", "stereo"):
            return layout  # type: ignore[return-value]
    if "person_keypoints" in data:
        return "mono"
    if "left_person_keypoints" in data:
        return "stereo"
    return None


def yolo_cache_status(
    path: Path,
    expected_frame_count: int,
    layout: YoloCacheLayout,
) -> YoloCacheStatus:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as data:
            file_layout = _layout_from_npz(data)
            if file_layout is None:
                return "missing"
            if file_layout != layout:
                return "wrong_layout"
            frame_count = int(data["frame_count"])
    except (OSError, KeyError, TypeError, ValueError):
        return "missing"
    if expected_frame_count > 0 and frame_count != expected_frame_count:
        return "stale"
    return "ready"


def yolo_cache_status_label(
    status: YoloCacheStatus,
    *,
    cached_frames: int = 0,
    expected_frames: int = 0,
) -> str:
    if status == "missing":
        return "Pose cache: not run"
    if status == "wrong_layout":
        return "Pose cache: wrong format — re-run"
    if status == "stale":
        return (
            f"Pose cache: stale (cached {cached_frames:,}, "
            f"video {expected_frames:,}) — re-run"
        )
    return f"Pose cache: ready ({cached_frames:,} frames)"


def save_mono_yolo_inferences(path: Path, store: MonoYoloInferenceStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(
        tmp,
        layout="mono",
        frame_count=store.frame_count,
        person_keypoints=store.person_keypoints,
        sides=store.sides,
    )
    tmp.replace(path)


def save_stereo_yolo_inferences(path: Path, store: StereoYoloInferenceStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(
        tmp,
        layout="stereo",
        frame_count=store.frame_count,
        left_person_keypoints=store.left_person_keypoints,
        left_sides=store.left_sides,
        right_person_keypoints=store.right_person_keypoints,
        right_sides=store.right_sides,
    )
    tmp.replace(path)


def save_yolo_inferences(path: Path, store: StereoYoloInferenceStore) -> None:
    save_stereo_yolo_inferences(path, store)


def load_mono_yolo_inferences(path: Path) -> MonoYoloInferenceStore:
    with np.load(path, allow_pickle=False) as data:
        if _layout_from_npz(data) != "mono":
            raise ValueError(f"Expected mono YOLO cache at {path}")
        return MonoYoloInferenceStore(
            frame_count=int(data["frame_count"]),
            person_keypoints=data["person_keypoints"],
            sides=data["sides"],
        )


def load_stereo_yolo_inferences(path: Path) -> StereoYoloInferenceStore:
    with np.load(path, allow_pickle=False) as data:
        if _layout_from_npz(data) != "stereo":
            raise ValueError(f"Expected stereo YOLO cache at {path}")
        return StereoYoloInferenceStore(
            frame_count=int(data["frame_count"]),
            left_person_keypoints=data["left_person_keypoints"],
            left_sides=data["left_sides"],
            right_person_keypoints=data["right_person_keypoints"],
            right_sides=data["right_sides"],
        )


def load_yolo_inferences(path: Path) -> StereoYoloInferenceStore:
    return load_stereo_yolo_inferences(path)


def populate_mono_pose_cache(store: MonoYoloInferenceStore, cache: StreamPlaybackCache) -> None:
    for frame_index in range(store.frame_count):
        cache.put_pose(frame_index, store.detection(frame_index))


def populate_stereo_pose_cache(store: StereoYoloInferenceStore, cache: PlaybackCache) -> None:
    for frame_index in range(store.frame_count):
        cache.main.put_pose(frame_index, store.detection("left", frame_index))
        cache.secondary.put_pose(frame_index, store.detection("right", frame_index))


def populate_pose_cache(store: StereoYoloInferenceStore, cache: PlaybackCache) -> None:
    populate_stereo_pose_cache(store, cache)


def try_load_pose_cache(
    path: Path,
    expected_frame_count: int,
    layout: YoloCacheLayout,
    cache: PlaybackCache,
) -> bool:
    if yolo_cache_status(path, expected_frame_count, layout) != "ready":
        return False
    if layout == "mono":
        store = load_mono_yolo_inferences(path)
        populate_mono_pose_cache(store, cache.main)
    else:
        store = load_stereo_yolo_inferences(path)
        populate_stereo_pose_cache(store, cache)
    return True


def _detection_to_row(
    detection: DominantHandDetection | None,
) -> tuple[np.ndarray, int]:
    if detection is None:
        return np.full((17, 3), np.nan, dtype=np.float32), -1
    return detection.person_keypoints.astype(np.float32), _side_to_int(detection.hand.side)


def open_stereo_timeline_reader(
    left_path: Path,
    right_path: Path,
    *,
    frame_count: int,
    fps: float,
) -> tuple[StereoFrameReader, int]:
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        raise OSError(f"Could not open videos:\n{left_path}\n{right_path}")

    left_total = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    right_total = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    effective_fps = fps if fps > 0 else float(left_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    timeline = load_stereo_timeline_for_videos(
        left_path,
        left_frame_count=left_total,
        right_frame_count=right_total,
        fps=effective_fps,
    )
    total = frame_count if frame_count > 0 else timeline.master_count
    reader = StereoFrameReader(left_cap, right_cap, timeline)
    return reader, total


def run_mono_yolo_inference_phase(
    *,
    video_path: Path,
    output_path: Path,
    frame_count: int,
    batch_size: int = YOLO_BATCH_SIZE,
    progress: YoloProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> MonoYoloInferenceStore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video:\n{video_path}")

    total = frame_count if frame_count > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    person = _empty_person_array(total)
    sides = _empty_sides_array(total)

    detector = PoseDetector()
    start = time.monotonic()

    try:
        for batch_start in range(0, total, batch_size):
            if cancel_check is not None and cancel_check():
                break

            batch_end = min(batch_start + batch_size, total)
            frames: list[np.ndarray] = []
            for frame_index in range(batch_start, batch_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)

            if not frames:
                break

            detections = detector.detect_batch(frames)
            for offset, (frame, people) in enumerate(zip(frames, detections)):
                frame_index = batch_start + offset
                detection = select_dominant_hand_detection(frame, people)
                person[frame_index], sides[frame_index] = _detection_to_row(detection)

                if progress is not None:
                    progress(
                        YoloBatchProgress(
                            frame_index=frame_index,
                            frame_count=total,
                            elapsed_s=time.monotonic() - start,
                        )
                    )
    finally:
        cap.release()

    store = MonoYoloInferenceStore(
        frame_count=total,
        person_keypoints=person,
        sides=sides,
    )
    save_mono_yolo_inferences(output_path, store)
    return store


def run_stereo_yolo_inference_phase(
    *,
    left_path: Path,
    right_path: Path,
    output_path: Path,
    frame_count: int,
    fps: float = 0.0,
    batch_size: int = YOLO_BATCH_SIZE,
    progress: YoloProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> StereoYoloInferenceStore:
    reader, total = open_stereo_timeline_reader(
        left_path,
        right_path,
        frame_count=frame_count,
        fps=fps,
    )

    left_person = _empty_person_array(total)
    left_sides = _empty_sides_array(total)
    right_person = _empty_person_array(total)
    right_sides = _empty_sides_array(total)

    detector = PoseDetector()
    start = time.monotonic()

    try:
        for batch_start in range(0, total, batch_size):
            if cancel_check is not None and cancel_check():
                break

            batch_end = min(batch_start + batch_size, total)
            left_frames: list[np.ndarray] = []
            right_frames: list[np.ndarray] = []
            for master_index in range(batch_start, batch_end):
                left_frame, right_frame = reader.read_at_master(master_index)
                if left_frame is None or right_frame is None:
                    break
                left_frames.append(left_frame)
                right_frames.append(right_frame)

            if not left_frames:
                break

            left_batch = detector.detect_batch(left_frames)
            right_batch = detector.detect_batch(right_frames)

            for offset, (left_frame, right_frame, left_people, right_people) in enumerate(
                zip(left_frames, right_frames, left_batch, right_batch)
            ):
                frame_index = batch_start + offset
                left_detection = select_dominant_hand_detection(left_frame, left_people)
                right_detection = select_dominant_hand_detection(right_frame, right_people)
                left_person[frame_index], left_sides[frame_index] = _detection_to_row(
                    left_detection
                )
                right_person[frame_index], right_sides[frame_index] = _detection_to_row(
                    right_detection
                )

                if progress is not None:
                    progress(
                        YoloBatchProgress(
                            frame_index=frame_index,
                            frame_count=total,
                            elapsed_s=time.monotonic() - start,
                        )
                    )
    finally:
        reader.release()

    store = StereoYoloInferenceStore(
        frame_count=total,
        left_person_keypoints=left_person,
        left_sides=left_sides,
        right_person_keypoints=right_person,
        right_sides=right_sides,
    )
    save_stereo_yolo_inferences(output_path, store)
    return store


def run_yolo_inference_phase(
    *,
    left_path: Path,
    right_path: Path,
    output_path: Path,
    frame_count: int,
    batch_size: int = YOLO_BATCH_SIZE,
    progress: YoloProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> StereoYoloInferenceStore:
    return run_stereo_yolo_inference_phase(
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        frame_count=frame_count,
        batch_size=batch_size,
        progress=progress,
        cancel_check=cancel_check,
    )
