from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from throw_detection.inference import ThrowInference
from trajectory_tracking import Phase, TrajectoryTracker
from trajectory_tracking.stereo import reconcile_stereo_trackers
from trajectory_tracking.release import (
    find_release_point_from_cache,
    find_secondary_release_at_frame,
)
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.filters import _extract_wrist_pos, wrist_pos_from_frame
from video_viewer.stereo_ball_detection import detect_stereo_balls
from video_viewer.stereo_playback import tracker_throw_label_during_left_hold
from video_viewer.stereo_timeline import StereoTimeline

from calibration import TableCalibration

from .config import GAME_JSON
from .game_data import GameSession, Point2D, ThrowRecord, new_game_session, save_game
from .triangulation import TriangulationResult, triangulate_throw


@dataclass
class ProcessorState:
    throw_count: int = 0
    last_speed_m_s: float | None = None


class GameTrackingProcessor:
    """Stereo throw + ball tracking with 3D triangulation and JSON export."""

    def __init__(self) -> None:
        self._throw_inference: ThrowInference | None = None
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
        self._last_paired_main_completion_id = 0
        self._last_paired_secondary_completion_id = 0
        self._main_phase_prev = Phase.DETECTING_THROW
        self._secondary_phase_prev = Phase.DETECTING_THROW
        self._next_throw_id = 1
        self._frame_size: tuple[int, int] | None = None
        self._fps: float = 30.0
        self._auto_persist = True

        self.state = ProcessorState()

    def set_calibration(self, calibration: TableCalibration | None) -> None:
        self._calibration = calibration

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
        if self._throw_inference is not None:
            self._throw_inference.reset()
        self._main_tracker.reset()
        self._secondary_tracker.reset()
        self._main_motion.reset()
        self._secondary_motion.reset()
        self._main_pending = []
        self._secondary_pending = []
        self._last_paired_main_completion_id = 0
        self._last_paired_secondary_completion_id = 0
        self._main_phase_prev = Phase.DETECTING_THROW
        self._secondary_phase_prev = Phase.DETECTING_THROW

    def reset(self) -> None:
        self.reset_tracking()
        self._session = None
        self._stereo_timeline = None
        self.state = ProcessorState()

    def _ensure_throw_inference(self) -> ThrowInference | None:
        if self._throw_inference is not None:
            return self._throw_inference
        if THROW_MODEL_PATH is None or not THROW_MODEL_PATH.is_file():
            return None
        self._throw_inference = ThrowInference(THROW_MODEL_PATH)
        return self._throw_inference

    def _maybe_reset_pending_on_scan(self, phase: Phase, prev: Phase, camera: str) -> Phase:
        if phase == Phase.SCANNING_BALL and prev != Phase.SCANNING_BALL:
            if camera == "main":
                self._main_pending = []
            else:
                self._secondary_pending = []
        return phase

    def _capture_detection(
        self,
        phase: Phase,
        detected_pos: tuple[int, int] | None,
        frame_index: int,
        pending: list[Point2D],
    ) -> None:
        if phase == Phase.TRACKING_BALL and detected_pos is not None:
            pending.append(Point2D(frame=frame_index, x=detected_pos[0], y=detected_pos[1]))

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

    def _try_pair_throw(self, cache: object | None = None) -> None:
        main_id = self._main_tracker._completion_id
        secondary_id = self._secondary_tracker._completion_id
        if (
            main_id <= self._last_paired_main_completion_id
            or secondary_id <= self._last_paired_secondary_completion_id
        ):
            return

        left_track = list(self._main_pending)
        right_track = list(self._secondary_pending)
        self._last_paired_main_completion_id = main_id
        self._last_paired_secondary_completion_id = secondary_id
        self._main_pending = []
        self._secondary_pending = []

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
            )
            if release is not None:
                left_track = [
                    Point2D(frame=release.frame, x=release.x, y=release.y),
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

        inference = self._ensure_throw_inference()
        if inference is None:
            return main_frame.copy(), secondary_frame.copy()

        prediction = inference.predict(
            main_frame,
            warmup_frames=main_warmup_frames,
            warmup_start_index=main_warmup_start_index,
            cache=cache.main if cache is not None else None,
            frame_index=frame_index,
        )
        throw_label = tracker_throw_label_during_left_hold(
            prediction.label,
            timeline=self._stereo_timeline,
            master_index=frame_index,
            cache=cache.main if cache is not None else None,
        )

        main_wrist_pos = _extract_wrist_pos(prediction.detection)
        secondary_wrist_pos: tuple[int, int] | None = None
        if throw_label == 1 or self._secondary_tracker.phase == Phase.SCANNING_BALL:
            secondary_wrist_pos = wrist_pos_from_frame(
                secondary_frame,
                cache=cache.secondary if cache is not None else None,
                frame_index=frame_index,
            )

        main_result = self._main_tracker.update(
            throw_label=throw_label,
            wrist_pos=main_wrist_pos,
            motion_mask=detection.main.motion_mask,
            alternate_motion_mask=detection.main.alternate_motion_mask,
            defer_detecting_throw=True,
            frame_index=frame_index,
        )
        secondary_result = self._secondary_tracker.update_secondary(
            throw_label=throw_label,
            wrist_pos=secondary_wrist_pos,
            motion_mask=detection.secondary.motion_mask,
            alternate_motion_mask=detection.secondary.alternate_motion_mask,
            defer_detecting_throw=True,
            frame_index=frame_index,
        )
        main_reconciled, secondary_reconciled = reconcile_stereo_trackers(
            self._main_tracker,
            self._secondary_tracker,
            throw_label=throw_label,
            wrist_pos=main_wrist_pos,
            secondary_wrist_pos=secondary_wrist_pos,
        )
        if main_reconciled:
            self._main_pending = []
        if secondary_reconciled:
            self._secondary_pending = []

        main_phase = self._main_tracker.phase
        secondary_phase = self._secondary_tracker.phase

        self._main_phase_prev = self._maybe_reset_pending_on_scan(
            main_phase, self._main_phase_prev, "main"
        )
        self._secondary_phase_prev = self._maybe_reset_pending_on_scan(
            secondary_phase, self._secondary_phase_prev, "secondary"
        )
        self._main_phase_prev = main_phase
        self._secondary_phase_prev = secondary_phase

        self._capture_detection(
            main_phase,
            main_result.detected_ball_pos,
            frame_index,
            self._main_pending,
        )
        self._capture_detection(
            secondary_phase,
            secondary_result.detected_ball_pos,
            frame_index,
            self._secondary_pending,
        )

        self._try_pair_throw(cache=cache.main if cache is not None else None)

        return main_frame, secondary_frame
