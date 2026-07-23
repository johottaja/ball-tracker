from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from pose_detection import PoseDetector, select_player_slot_detections
from pose_detection.types import PlayerSide
from throw_detection.inference import ThrowInference, ThrowPrediction
from trajectory_tracking import Phase, TrajectoryTracker
from trajectory_tracking.stereo import reconcile_stereo_trackers
from trajectory_tracking.release import (
    find_release_point_from_cache,
    find_secondary_release_at_frame,
)
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.filters import _extract_wrist_pos
from video_viewer.stereo_ball_detection import detect_stereo_balls
from video_viewer.stereo_timeline import StereoTimeline
from video_viewer.stereo_playback import tracker_throw_label_during_hold

from calibration import TableCalibration, infer_stereo_screen_side_mapping
from trajectory_tracking.config import SECTOR_DIRECTION_DEG

from .config import GAME_JSON
from .game_data import GameSession, Point2D, ThrowRecord, new_game_session, save_game
from .triangulation import TriangulationResult, triangulate_throw


@dataclass
class ProcessorState:
    throw_count: int = 0
    last_speed_m_s: float | None = None


@dataclass(frozen=True)
class CompletedTrack:
    thrower_side: PlayerSide
    completion_id: int
    points: list[Point2D]

    @property
    def time_range(self) -> tuple[float, float] | None:
        times = [point.time_s for point in self.points if point.time_s is not None]
        return (min(times), max(times)) if times else None


class GameTrackingProcessor:
    """Stereo throw + ball tracking with 3D triangulation and JSON export."""

    def __init__(self) -> None:
        self._throw_inferences: dict[PlayerSide, ThrowInference] = {}
        self._pose_detector = PoseDetector()
        self._main_tracker = TrajectoryTracker()
        self._secondary_tracker = TrajectoryTracker()
        self._main_motion = MotionMaskBuilder()
        self._secondary_motion = MotionMaskBuilder()
        self._calibration: TableCalibration | None = None
        self._stereo_timeline: StereoTimeline | None = None
        self._session: GameSession | None = None
        self._game_json_path = GAME_JSON
        self._on_throw_recorded: Callable[[ThrowRecord], None] | None = None

        self._main_pending: list[Point2D] = []
        self._secondary_pending: list[Point2D] = []
        self._main_completed: list[CompletedTrack] = []
        self._secondary_completed: list[CompletedTrack] = []
        self._last_seen_main_completion_id = 0
        self._last_seen_secondary_completion_id = 0
        self._active_thrower_side: PlayerSide | None = None
        self._secondary_side_for_main: dict[PlayerSide, PlayerSide] | None = None
        self._next_throw_id = 1
        self._frame_size: tuple[int, int] | None = None
        self._fps: float = 30.0
        self._auto_persist = True

        self.state = ProcessorState()

    def set_calibration(self, calibration: TableCalibration | None) -> None:
        self._calibration = calibration
        mapping = (
            infer_stereo_screen_side_mapping(calibration)
            if calibration is not None
            else None
        )
        if mapping is None:
            self._secondary_side_for_main = None
            if calibration is not None:
                print(
                    "Secondary player-side mapping is ambiguous; "
                    "using full-frame secondary ball scans"
                )
            return
        self._secondary_side_for_main = {
            "left": mapping.main_left_to_secondary,
            "right": "right"
            if mapping.main_left_to_secondary == "left"
            else "left",
        }

    def set_stereo_timeline(self, timeline: StereoTimeline | None) -> None:
        self._stereo_timeline = timeline

    def set_on_throw_recorded(self, callback: Callable[[ThrowRecord], None] | None) -> None:
        self._on_throw_recorded = callback

    def set_ball_detection_method(self, method: BallDetectionMethod) -> None:
        self._main_motion.set_method(method)
        self._secondary_motion.set_method(method)

    def set_auto_persist(self, enabled: bool) -> None:
        self._auto_persist = enabled

    def flush_session(self) -> None:
        self._persist_session(force=True)

    def throw_buffer_size(self) -> int:
        inference = self._ensure_throw_inference()
        if inference is None:
            from throw_detection.config import BUFFER_SIZE

            return BUFFER_SIZE
        return inference.buffer_size

    def begin_session(
        self,
        *,
        fps: float,
        frame_count: int,
        game_json_path: Path = GAME_JSON,
    ) -> None:
        self.reset_tracking()
        self._game_json_path = game_json_path
        self._session = new_game_session(
            fps=fps,
            frame_count=frame_count,
            calibration=self._calibration,
        )
        self._next_throw_id = 1
        self.state = ProcessorState()
        self._persist_session()

    def load_session(self, session: GameSession, *, game_json_path: Path = GAME_JSON) -> None:
        self.reset_tracking()
        self._game_json_path = game_json_path
        self._session = session
        throws = session.throws or []
        self._next_throw_id = max((t.id for t in throws), default=0) + 1
        self.state = ProcessorState(
            throw_count=len(throws),
            last_speed_m_s=throws[-1].speed_m_s if throws else None,
        )

    def reset_tracking(self) -> None:
        for inference in self._throw_inferences.values():
            inference.reset()
        self._main_tracker.reset()
        self._secondary_tracker.reset()
        self._main_motion.reset()
        self._secondary_motion.reset()
        self._main_pending = []
        self._secondary_pending = []
        self._main_completed = []
        self._secondary_completed = []
        self._last_seen_main_completion_id = 0
        self._last_seen_secondary_completion_id = 0
        self._active_thrower_side = None

    def reset(self) -> None:
        self.reset_tracking()
        self._session = None
        self._stereo_timeline = None
        self.state = ProcessorState()

    def _ensure_throw_inference(
        self, thrower_side: PlayerSide = "right"
    ) -> ThrowInference | None:
        if thrower_side in self._throw_inferences:
            return self._throw_inferences[thrower_side]
        if THROW_MODEL_PATH is None or not THROW_MODEL_PATH.is_file():
            return None
        inference = ThrowInference(
            THROW_MODEL_PATH,
            detector=self._pose_detector,
            mirror_x=thrower_side == "left",
        )
        self._throw_inferences[thrower_side] = inference
        return inference

    def _player_predictions(
        self,
        frame: np.ndarray,
        *,
        cache: object | None,
        frame_index: int,
    ) -> dict[PlayerSide, ThrowPrediction]:
        if cache is not None and all(
            cache.main.has_player_gru(player_side, frame_index)
            for player_side in ("left", "right")
        ):
            return {
                player_side: cache.main.get_player_gru(player_side, frame_index)
                for player_side in ("left", "right")
            }

        people = self._pose_detector.detect(frame)
        detections = select_player_slot_detections(frame, people)
        predictions: dict[PlayerSide, ThrowPrediction] = {}
        for player_side in ("left", "right"):
            inference = self._ensure_throw_inference(player_side)
            if inference is None:
                continue
            predictions[player_side] = inference.predict_from_detection(
                detections[player_side]
            )
        return predictions

    def _active_prediction(
        self,
        predictions: dict[PlayerSide, ThrowPrediction],
    ) -> tuple[PlayerSide | None, ThrowPrediction | None]:
        if self._active_thrower_side is not None:
            return (
                self._active_thrower_side,
                predictions.get(self._active_thrower_side),
            )
        candidates = [
            (player_side, prediction)
            for player_side, prediction in predictions.items()
            if prediction.label == 1
        ]
        if not candidates:
            return None, None
        return max(candidates, key=lambda item: item[1].probability)

    @staticmethod
    def _sector_direction_for_side(player_side: PlayerSide) -> float:
        return (
            SECTOR_DIRECTION_DEG
            if player_side == "right"
            else (180.0 - SECTOR_DIRECTION_DEG) % 360.0
        )

    def _capture_time(self, side: Literal["left", "right"], frame_index: int) -> float:
        if self._stereo_timeline is not None:
            return self._stereo_timeline.capture_time(side, frame_index)
        return frame_index / self._fps

    def _capture_detection(
        self,
        phase: Phase,
        detected_pos: tuple[int, int] | None,
        frame_index: int,
        pending: list[Point2D],
        *,
        side: Literal["left", "right"],
    ) -> None:
        if phase != Phase.TRACKING_BALL or detected_pos is None:
            return
        if self._stereo_timeline is not None and self._stereo_timeline.is_hold(side, frame_index):
            return
        pending.append(
            Point2D(
                frame=frame_index,
                x=detected_pos[0],
                y=detected_pos[1],
                time_s=self._capture_time(side, frame_index),
            )
        )

    def _log_triangulation_result(
        self,
        result: TriangulationResult,
        *,
        main_completion_id: int,
        secondary_completion_id: int,
    ) -> None:
        if result.ok and result.throw is not None:
            throw = result.throw
            speed = throw.speed_m_s
            speed_text = f"{speed:.2f} m/s" if speed is not None else "unknown speed"
            print(
                "Triangulated throw "
                f"{throw.id}: frames {throw.start_frame}-{throw.end_frame}, "
                f"{len(throw.points_3d)} 3D points, {speed_text}"
            )
            return

        print(
            "Triangulation failed "
            f"(left completion {main_completion_id}, "
            f"right completion {secondary_completion_id}): "
            f"{result.error or 'unknown error'}"
        )

    def _queue_completed_tracks(self) -> None:
        thrower_side = self._active_thrower_side
        if thrower_side is None:
            return
        main_id = self._main_tracker._completion_id
        if main_id > self._last_seen_main_completion_id:
            self._main_completed.append(
                CompletedTrack(thrower_side, main_id, list(self._main_pending))
            )
            self._main_pending = []
            self._last_seen_main_completion_id = main_id
        secondary_id = self._secondary_tracker._completion_id
        if secondary_id > self._last_seen_secondary_completion_id:
            self._secondary_completed.append(
                CompletedTrack(thrower_side, secondary_id, list(self._secondary_pending))
            )
            self._secondary_pending = []
            self._last_seen_secondary_completion_id = secondary_id

    @staticmethod
    def _tracks_overlap(main: CompletedTrack, secondary: CompletedTrack) -> bool:
        main_range = main.time_range
        secondary_range = secondary.time_range
        if main_range is None or secondary_range is None:
            return bool(main.points and secondary.points)
        return max(main_range[0], secondary_range[0]) <= min(
            main_range[1], secondary_range[1]
        )

    def _try_pair_throw(self, cache: object | None = None) -> None:
        match: tuple[CompletedTrack, CompletedTrack] | None = None
        for main in self._main_completed:
            for secondary in self._secondary_completed:
                if (
                    main.thrower_side == secondary.thrower_side
                    and self._tracks_overlap(main, secondary)
                ):
                    match = (main, secondary)
                    break
            if match is not None:
                break
        if match is None:
            return
        main_track, secondary_track = match
        self._main_completed.remove(main_track)
        self._secondary_completed.remove(secondary_track)
        main_id = main_track.completion_id
        secondary_id = secondary_track.completion_id
        left_track = list(main_track.points)
        right_track = list(secondary_track.points)

        if self._frame_size is None:
            print(
                "Triangulation failed "
                f"(left completion {main_id}, right completion {secondary_id}): "
                "frame size not set"
            )
            return

        if cache is not None:
            release = find_release_point_from_cache(
                left_track,
                self._main_tracker._completed_parabola_fit,
                cache,
                player_side=main_track.thrower_side,
            )
            if release is not None:
                left_track = [
                    Point2D(
                        frame=release.frame,
                        x=release.x,
                        y=release.y,
                        time_s=self._capture_time("left", release.frame),
                    ),
                    *left_track,
                ]
                secondary_release = find_secondary_release_at_frame(
                    right_track,
                    self._secondary_tracker._completed_parabola_fit,
                    release.frame,
                    timeline_offset=0.0,
                )
                if secondary_release is not None:
                    right_track = [
                        Point2D(
                            frame=secondary_release.frame,
                            x=secondary_release.x,
                            y=secondary_release.y,
                            time_s=self._capture_time("right", secondary_release.frame),
                        ),
                        *right_track,
                    ]

        result = triangulate_throw(
            left_track,
            right_track,
            calibration=self._calibration,
            frame_size=self._frame_size,
            fps=self._fps,
            throw_id=self._next_throw_id,
            thrower_side=main_track.thrower_side,
            timeline=self._stereo_timeline,
        )
        self._log_triangulation_result(
            result,
            main_completion_id=main_id,
            secondary_completion_id=secondary_id,
        )
        if not result.ok or result.throw is None:
            return

        throw = result.throw
        self._next_throw_id += 1
        if self._session is not None:
            self._session.throws = self._session.throws or []
            self._session.throws.append(throw)
            self._persist_session()

        self.state.throw_count += 1
        self.state.last_speed_m_s = throw.speed_m_s
        if self._on_throw_recorded is not None:
            self._on_throw_recorded(throw)

    def _persist_session(self, *, force: bool = False) -> None:
        if self._session is not None and (force or self._auto_persist):
            save_game(self._game_json_path, self._session)

    def apply(
        self,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
        *,
        frame_index: int,
        main_warmup_frames: list[np.ndarray] | None = None,
        main_warmup_start_index: int | None = None,
        main_previous_frame: np.ndarray | None = None,
        main_next_frame: np.ndarray | None = None,
        main_mog2_warmup_frames: list[np.ndarray] | None = None,
        secondary_previous_frame: np.ndarray | None = None,
        secondary_next_frame: np.ndarray | None = None,
        secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
        cache: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if video_fps is not None and video_fps > 0:
            self._fps = video_fps

        h, w = main_frame.shape[:2]
        self._frame_size = (w, h)

        if main_warmup_frames is not None:
            self.reset_tracking()

        detection = detect_stereo_balls(
            self._main_motion,
            self._secondary_motion,
            main_frame,
            secondary_frame,
            main_previous_frame=main_previous_frame,
            main_next_frame=main_next_frame,
            main_mog2_warmup_frames=main_mog2_warmup_frames,
            secondary_previous_frame=secondary_previous_frame,
            secondary_next_frame=secondary_next_frame,
            secondary_mog2_warmup_frames=secondary_mog2_warmup_frames,
            main_cache=cache.main if cache is not None else None,
            secondary_cache=cache.secondary if cache is not None else None,
            frame_index=frame_index,
        )

        predictions = self._player_predictions(
            main_frame,
            cache=cache,
            frame_index=frame_index,
        )
        if not predictions:
            return main_frame.copy(), secondary_frame.copy()

        thrower_side, prediction = self._active_prediction(predictions)
        if self._active_thrower_side is None and thrower_side is not None:
            self._active_thrower_side = thrower_side
        active_side = self._active_thrower_side
        if active_side is not None:
            prediction = predictions.get(active_side)
        throw_label = prediction.label if prediction is not None else 0
        throw_label = tracker_throw_label_during_hold(
            throw_label,
            side="left",
            timeline=self._stereo_timeline,
            master_index=frame_index,
            cache=cache.main if cache is not None else None,
        )
        main_wrist_pos = (
            _extract_wrist_pos(prediction.detection) if prediction is not None else None
        )
        if active_side is not None:
            self._main_tracker.set_sector_direction_deg(
                self._sector_direction_for_side(active_side)
            )

        secondary_wrist_pos: tuple[int, int] | None = None
        secondary_side = (
            self._secondary_side_for_main[active_side]
            if active_side is not None and self._secondary_side_for_main is not None
            else None
        )
        if secondary_side is not None:
            self._secondary_tracker.set_sector_direction_deg(
                self._sector_direction_for_side(secondary_side)
            )
        if (
            secondary_side is not None
            and (throw_label == 1 or self._secondary_tracker.phase == Phase.SCANNING_BALL)
        ):
            if (
                cache is not None
                and cache.secondary.has_player_pose(secondary_side, frame_index)
            ):
                secondary_detection = cache.secondary.get_player_pose(
                    secondary_side, frame_index
                )
            else:
                secondary_detection = select_player_slot_detections(
                    secondary_frame,
                    self._pose_detector.detect(secondary_frame),
                )[secondary_side]
            secondary_wrist_pos = _extract_wrist_pos(secondary_detection)

        main_fresh = self._stereo_timeline is None or not self._stereo_timeline.is_hold("left", frame_index)
        secondary_fresh = self._stereo_timeline is None or not self._stereo_timeline.is_hold("right", frame_index)
        main_result = (
            self._main_tracker.update(
                throw_label=throw_label,
                wrist_pos=main_wrist_pos,
                motion_mask=detection.main.motion_mask,
                alternate_motion_mask=detection.main.alternate_motion_mask,
                defer_detecting_throw=True,
                frame_index=frame_index,
            )
            if main_fresh
            else self._main_tracker.snapshot()
        )
        secondary_result = (
            self._secondary_tracker.update_secondary(
                throw_label=throw_label,
                wrist_pos=secondary_wrist_pos,
                motion_mask=detection.secondary.motion_mask,
                alternate_motion_mask=detection.secondary.alternate_motion_mask,
                defer_detecting_throw=True,
                frame_index=frame_index,
            )
            if secondary_fresh
            else self._secondary_tracker.snapshot()
        )
        if main_fresh and secondary_fresh:
            main_reconciled, secondary_reconciled = reconcile_stereo_trackers(
                self._main_tracker,
                self._secondary_tracker,
                throw_label=throw_label,
                wrist_pos=main_wrist_pos,
                secondary_wrist_pos=secondary_wrist_pos,
            )
        else:
            main_reconciled = secondary_reconciled = False
        if main_reconciled:
            self._main_pending = []
        if secondary_reconciled:
            self._secondary_pending = []

        main_phase = self._main_tracker.phase
        secondary_phase = self._secondary_tracker.phase

        self._capture_detection(
            main_phase,
            main_result.detected_ball_pos,
            frame_index,
            self._main_pending,
            side="left",
        )
        self._capture_detection(
            secondary_phase,
            secondary_result.detected_ball_pos,
            frame_index,
            self._secondary_pending,
            side="right",
        )

        self._queue_completed_tracks()
        self._try_pair_throw(cache=cache.main if cache is not None else None)
        if (
            self._active_thrower_side is not None
            and self._main_tracker.phase == Phase.DETECTING_THROW
            and self._secondary_tracker.phase == Phase.DETECTING_THROW
        ):
            self._active_thrower_side = None

        return main_frame, secondary_frame
