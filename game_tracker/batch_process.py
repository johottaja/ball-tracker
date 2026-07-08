from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from calibration import TableCalibration
from pose_detection import (
    DominantHandDetection,
    PoseDetector,
    dominant_hand_detection_from_keypoints,
    select_dominant_hand_detection,
)
from pose_detection.types import HandSide
from video_viewer.ball_motion import (
    BallDetectionMethod,
    uses_frame_diff_component,
)
from video_viewer.playback_cache import PlaybackCache

from .config import RECORDINGS_DIR, YOLO_BATCH_SIZE
from .processor import GameTrackingProcessor


@dataclass(frozen=True)
class BatchProgress:
    phase: Literal["yolo", "tracking"]
    phase_title: str
    frame_index: int
    frame_count: int
    elapsed_s: float
    throws: int = 0

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


ProgressCallback = Callable[[BatchProgress], None]


def yolo_inferences_path(game_stem: str) -> Path:
    return RECORDINGS_DIR / f"yolo_inferences-{game_stem}.npz"


def _side_to_int(side: HandSide | None) -> int:
    if side is None:
        return -1
    return 0 if side == "left" else 1


def _int_to_side(value: int) -> HandSide | None:
    if value < 0:
        return None
    return "left" if value == 0 else "right"


@dataclass(frozen=True)
class YoloInferenceStore:
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


def _empty_person_array(count: int) -> np.ndarray:
    return np.full((count, 17, 3), np.nan, dtype=np.float32)


def _empty_sides_array(count: int) -> np.ndarray:
    return np.full(count, -1, dtype=np.int8)


def save_yolo_inferences(path: Path, store: YoloInferenceStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(
        tmp,
        frame_count=store.frame_count,
        left_person_keypoints=store.left_person_keypoints,
        left_sides=store.left_sides,
        right_person_keypoints=store.right_person_keypoints,
        right_sides=store.right_sides,
    )
    tmp.replace(path)


def load_yolo_inferences(path: Path) -> YoloInferenceStore:
    with np.load(path, allow_pickle=False) as data:
        return YoloInferenceStore(
            frame_count=int(data["frame_count"]),
            left_person_keypoints=data["left_person_keypoints"],
            left_sides=data["left_sides"],
            right_person_keypoints=data["right_person_keypoints"],
            right_sides=data["right_sides"],
        )


def populate_pose_cache(store: YoloInferenceStore, cache: PlaybackCache) -> None:
    for frame_index in range(store.frame_count):
        cache.main.put_pose(frame_index, store.detection("left", frame_index))
        cache.secondary.put_pose(frame_index, store.detection("right", frame_index))


def _read_stereo_pair(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    ok_left, left_frame = left_cap.read()
    ok_right, right_frame = right_cap.read()
    if not ok_left or left_frame is None or not ok_right or right_frame is None:
        return False, None, None
    return True, left_frame, right_frame


def _open_stereo_caps(left_path: Path, right_path: Path) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        raise OSError(f"Could not open videos:\n{left_path}\n{right_path}")
    return left_cap, right_cap


def _frame_count_from_caps(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
    frame_count: int,
) -> int:
    if frame_count > 0:
        return frame_count
    left_total = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    right_total = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if left_total and right_total:
        return min(left_total, right_total)
    return max(left_total, right_total)


def _iter_stereo_frames_sequential(
    left_cap: cv2.VideoCapture,
    right_cap: cv2.VideoCapture,
    total: int,
    *,
    need_neighbors: bool,
):
    """Yield stereo frames in order without per-frame video seeks."""
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ok, left_cur, right_cur = _read_stereo_pair(left_cap, right_cap)
    if not ok:
        return

    left_prev: np.ndarray | None = None
    right_prev: np.ndarray | None = None
    left_next: np.ndarray | None = None
    right_next: np.ndarray | None = None
    has_next = False

    if need_neighbors and total > 1:
        has_next, left_next, right_next = _read_stereo_pair(left_cap, right_cap)

    for frame_index in range(total):
        if not ok or left_cur is None or right_cur is None:
            break

        yield (
            frame_index,
            left_cur,
            right_cur,
            left_prev,
            right_prev,
            left_next if has_next else None,
            right_next if has_next else None,
        )

        if not need_neighbors:
            ok, left_cur, right_cur = _read_stereo_pair(left_cap, right_cap)
            continue

        left_prev, right_prev = left_cur, right_cur
        left_cur, right_cur = left_next, right_next
        ok = has_next

        if frame_index + 2 < total:
            has_next, left_next, right_next = _read_stereo_pair(left_cap, right_cap)
        else:
            has_next = False
            left_next, right_next = None, None


def _detection_to_row(
    detection: DominantHandDetection | None,
) -> tuple[np.ndarray, int]:
    if detection is None:
        return np.full((17, 3), np.nan, dtype=np.float32), -1
    return detection.person_keypoints.astype(np.float32), _side_to_int(detection.hand.side)


def run_yolo_inference_phase(
    *,
    left_path: Path,
    right_path: Path,
    output_path: Path,
    frame_count: int,
    batch_size: int = YOLO_BATCH_SIZE,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> YoloInferenceStore:
    left_cap, right_cap = _open_stereo_caps(left_path, right_path)
    total = _frame_count_from_caps(left_cap, right_cap, frame_count)

    left_person = _empty_person_array(total)
    left_sides = _empty_sides_array(total)
    right_person = _empty_person_array(total)
    right_sides = _empty_sides_array(total)

    detector = PoseDetector()
    start = time.monotonic()

    try:
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for batch_start in range(0, total, batch_size):
            if cancel_check is not None and cancel_check():
                break

            batch_end = min(batch_start + batch_size, total)
            left_frames: list[np.ndarray] = []
            right_frames: list[np.ndarray] = []
            for _ in range(batch_end - batch_start):
                ok, left_frame, right_frame = _read_stereo_pair(left_cap, right_cap)
                if not ok or left_frame is None or right_frame is None:
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
                        BatchProgress(
                            phase="yolo",
                            phase_title="YOLO pose inference",
                            frame_index=frame_index,
                            frame_count=total,
                            elapsed_s=time.monotonic() - start,
                        )
                    )
    finally:
        left_cap.release()
        right_cap.release()

    store = YoloInferenceStore(
        frame_count=total,
        left_person_keypoints=left_person,
        left_sides=left_sides,
        right_person_keypoints=right_person,
        right_sides=right_sides,
    )
    save_yolo_inferences(output_path, store)
    return store


def run_tracking_phase(
    *,
    left_path: Path,
    right_path: Path,
    game_json_path: Path,
    yolo_store: YoloInferenceStore,
    ball_method: BallDetectionMethod,
    calibration: TableCalibration | None,
    fps: float,
    frame_count: int,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GameTrackingProcessor:
    processor = GameTrackingProcessor()
    processor.set_calibration(calibration)
    processor.set_ball_detection_method(ball_method)
    processor.set_auto_persist(False)
    processor.begin_session(fps=fps, frame_count=frame_count, game_json_path=game_json_path)

    left_cap, right_cap = _open_stereo_caps(left_path, right_path)

    cache = PlaybackCache()
    populate_pose_cache(yolo_store, cache)

    start = time.monotonic()
    total = yolo_store.frame_count
    need_neighbors = uses_frame_diff_component(ball_method)

    try:
        for (
            frame_index,
            left_frame,
            right_frame,
            left_previous,
            right_previous,
            left_next,
            right_next,
        ) in _iter_stereo_frames_sequential(
            left_cap,
            right_cap,
            total,
            need_neighbors=need_neighbors,
        ):
            if cancel_check is not None and cancel_check():
                break

            processor.apply(
                left_frame,
                right_frame,
                frame_index=frame_index,
                main_previous_frame=left_previous,
                main_next_frame=left_next,
                main_mog2_warmup_frames=None,
                secondary_previous_frame=right_previous,
                secondary_next_frame=right_next,
                secondary_mog2_warmup_frames=None,
                video_fps=fps,
                cache=cache,
            )

            if progress is not None:
                progress(
                    BatchProgress(
                        phase="tracking",
                        phase_title="Game tracking",
                        frame_index=frame_index,
                        frame_count=total,
                        elapsed_s=time.monotonic() - start,
                        throws=processor.state.throw_count,
                    )
                )
    finally:
        left_cap.release()
        right_cap.release()

    processor.flush_session()
    return processor


def process_game_recording(
    *,
    left_path: Path,
    right_path: Path,
    game_json_path: Path,
    ball_method: BallDetectionMethod,
    calibration: TableCalibration | None,
    fps: float,
    frame_count: int,
    progress: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GameTrackingProcessor:
    """Two-phase offline processing: YOLO inference, then game tracking."""
    yolo_path = yolo_inferences_path(game_json_path.stem)
    if yolo_path.is_file():
        yolo_store = load_yolo_inferences(yolo_path)
        if yolo_store.frame_count != frame_count and frame_count > 0:
            yolo_store = run_yolo_inference_phase(
                left_path=left_path,
                right_path=right_path,
                output_path=yolo_path,
                frame_count=frame_count,
                progress=progress,
                cancel_check=cancel_check,
            )
    else:
        yolo_store = run_yolo_inference_phase(
            left_path=left_path,
            right_path=right_path,
            output_path=yolo_path,
            frame_count=frame_count,
            progress=progress,
            cancel_check=cancel_check,
        )
    if cancel_check is not None and cancel_check():
        return GameTrackingProcessor()

    return run_tracking_phase(
        left_path=left_path,
        right_path=right_path,
        game_json_path=game_json_path,
        yolo_store=yolo_store,
        ball_method=ball_method,
        calibration=calibration,
        fps=fps,
        frame_count=frame_count,
        progress=progress,
        cancel_check=cancel_check,
    )
