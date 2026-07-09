from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from calibration import TableCalibration
from video_viewer.ball_motion import BallDetectionMethod, uses_frame_diff_component
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.gru_batch import (
    GruBatchProgress,
    GruInferenceStore,
    load_gru_inferences,
    populate_stereo_gru_cache,
    run_stereo_gru_inference_phase,
    save_gru_inferences,
    gru_cache_status,
)
from video_viewer.playback_cache import PlaybackCache
from video_viewer.stereo_playback import iter_stereo_timeline
from video_viewer.yolo_batch import (
    StereoYoloInferenceStore,
    YoloBatchProgress,
    load_stereo_yolo_inferences,
    open_stereo_timeline_reader,
    populate_stereo_pose_cache,
    run_stereo_yolo_inference_phase,
    yolo_cache_status,
)

from .config import GRU_INFERENCES, YOLO_INFERENCES
from .processor import GameTrackingProcessor

# Re-export shared batch APIs for callers that import from here.
__all__ = [
    "BatchProgress",
    "GameTrackingProcessor",
    "GruInferenceStore",
    "ProgressCallback",
    "YoloInferenceStore",
    "load_gru_inferences",
    "load_yolo_inferences",
    "populate_gru_cache",
    "populate_pose_cache",
    "process_game_recording",
    "run_gru_inference_phase",
    "run_tracking_phase",
    "run_yolo_inference_phase",
    "save_gru_inferences",
    "save_yolo_inferences",
]

load_yolo_inferences = load_stereo_yolo_inferences
populate_pose_cache = populate_stereo_pose_cache
populate_gru_cache = populate_stereo_gru_cache
run_yolo_inference_phase = run_stereo_yolo_inference_phase

YoloInferenceStore = StereoYoloInferenceStore
run_gru_inference_phase = run_stereo_gru_inference_phase


@dataclass(frozen=True)
class BatchProgress:
    phase: Literal["yolo", "gru", "tracking"]
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


def run_tracking_phase(
    *,
    left_path: Path,
    right_path: Path,
    game_json_path: Path,
    yolo_store: StereoYoloInferenceStore,
    gru_store: GruInferenceStore | None,
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

    reader, total = open_stereo_timeline_reader(
        left_path,
        right_path,
        frame_count=frame_count,
        fps=fps,
    )
    processor.set_stereo_timeline(reader.timeline)

    cache = PlaybackCache()
    populate_stereo_pose_cache(yolo_store, cache)
    if gru_store is not None:
        populate_stereo_gru_cache(gru_store, cache)

    start = time.monotonic()
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
        ) in iter_stereo_timeline(reader, need_neighbors=need_neighbors):
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
        reader.release()

    processor.flush_session()
    return processor


def _yolo_progress_adapter(
    callback: ProgressCallback | None,
) -> Callable[[YoloBatchProgress], None] | None:
    if callback is None:
        return None

    def on_progress(update: YoloBatchProgress) -> None:
        callback(
            BatchProgress(
                phase="yolo",
                phase_title="YOLO pose inference",
                frame_index=update.frame_index,
                frame_count=update.frame_count,
                elapsed_s=update.elapsed_s,
            )
        )

    return on_progress


def _gru_progress_adapter(
    callback: ProgressCallback | None,
) -> Callable[[GruBatchProgress], None] | None:
    if callback is None:
        return None

    def on_progress(update: GruBatchProgress) -> None:
        callback(
            BatchProgress(
                phase="gru",
                phase_title="GRU throw inference",
                frame_index=update.frame_index,
                frame_count=update.frame_count,
                elapsed_s=update.elapsed_s,
            )
        )

    return on_progress


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
    """Three-phase offline processing: YOLO, GRU, then game tracking."""
    yolo_progress = _yolo_progress_adapter(progress)
    gru_progress = _gru_progress_adapter(progress)

    if yolo_cache_status(YOLO_INFERENCES, frame_count, "stereo") == "ready":
        yolo_store = load_stereo_yolo_inferences(YOLO_INFERENCES)
    else:
        yolo_store = run_stereo_yolo_inference_phase(
            left_path=left_path,
            right_path=right_path,
            output_path=YOLO_INFERENCES,
            frame_count=frame_count,
            fps=fps,
            progress=yolo_progress,
            cancel_check=cancel_check,
        )
    if cancel_check is not None and cancel_check():
        return GameTrackingProcessor()

    gru_store: GruInferenceStore | None = None
    if THROW_MODEL_PATH is not None and THROW_MODEL_PATH.is_file():
        if gru_cache_status(GRU_INFERENCES, frame_count, THROW_MODEL_PATH) == "ready":
            gru_store = load_gru_inferences(GRU_INFERENCES)
        else:
            gru_store = run_stereo_gru_inference_phase(
                yolo_store=yolo_store,
                model_path=THROW_MODEL_PATH,
                output_path=GRU_INFERENCES,
                progress=gru_progress,
                cancel_check=cancel_check,
            )
    if cancel_check is not None and cancel_check():
        return GameTrackingProcessor()

    return run_tracking_phase(
        left_path=left_path,
        right_path=right_path,
        game_json_path=game_json_path,
        yolo_store=yolo_store,
        gru_store=gru_store,
        ball_method=ball_method,
        calibration=calibration,
        fps=fps,
        frame_count=frame_count,
        progress=progress,
        cancel_check=cancel_check,
    )
