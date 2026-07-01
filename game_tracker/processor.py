from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from throw_detection.inference import ThrowInference
from trajectory_tracking import Phase, TrajectoryTracker
from video_viewer.ball_motion import BallDetectionMethod, MotionMaskBuilder
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.filters import _extract_wrist_pos

from .config import GAME_JSON
from .game_data import GameSession, Point2D, ThrowRecord, new_game_session, save_game
from .setup_config import CameraSetup
from .triangulation import triangulate_throw


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
        self._setup = CameraSetup()
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

        self.state = ProcessorState()

    def set_camera_setup(self, setup: CameraSetup) -> None:
        self._setup = setup
        if self._session is not None:
            self._session.camera_setup = setup

    def set_on_throw_recorded(self, callback: Callable[[ThrowRecord], None] | None) -> None:
        self._on_throw_recorded = callback

    def set_ball_detection_method(self, method: BallDetectionMethod) -> None:
        self._main_motion.set_method(method)
        self._secondary_motion.set_method(method)

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
            camera_setup=self._setup,
        )
        self._next_throw_id = 1
        self.state = ProcessorState()
        self._persist_session()

    def load_session(self, session: GameSession, *, game_json_path: Path = GAME_JSON) -> None:
        self.reset_tracking()
        self._game_json_path = game_json_path
        self._session = session
        self._setup = session.camera_setup or self._setup
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
        self.state = ProcessorState()

    def _ensure_throw_inference(self) -> ThrowInference | None:
        if self._throw_inference is not None:
            return self._throw_inference
        if THROW_MODEL_PATH is None or not THROW_MODEL_PATH.is_file():
            return None
        self._throw_inference = ThrowInference(THROW_MODEL_PATH)
        return self._throw_inference

    def _sync_stereo_phases(self) -> None:
        main = self._main_tracker
        secondary = self._secondary_tracker
        if (
            main.phase == Phase.AWAITING_PARTNER
            and secondary.phase == Phase.AWAITING_PARTNER
        ):
            main.phase = Phase.DETECTING_THROW
            secondary.phase = Phase.DETECTING_THROW

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

    def _try_pair_throw(self) -> None:
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
            return

        throw = triangulate_throw(
            left_track,
            right_track,
            setup=self._setup,
            frame_size=self._frame_size,
            fps=self._fps,
            throw_id=self._next_throw_id,
        )
        if throw is None:
            return

        self._next_throw_id += 1
        if self._session is not None:
            self._session.throws = self._session.throws or []
            self._session.throws.append(throw)
            self._persist_session()

        self.state.throw_count += 1
        self.state.last_speed_m_s = throw.speed_m_s
        if self._on_throw_recorded is not None:
            self._on_throw_recorded(throw)

    def _persist_session(self) -> None:
        if self._session is not None:
            save_game(self._game_json_path, self._session)

    def apply(
        self,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
        *,
        frame_index: int,
        main_warmup_frames: list[np.ndarray] | None = None,
        main_previous_frame: np.ndarray | None = None,
        main_mog2_warmup_frames: list[np.ndarray] | None = None,
        secondary_previous_frame: np.ndarray | None = None,
        secondary_mog2_warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if video_fps is not None and video_fps > 0:
            self._fps = video_fps

        h, w = main_frame.shape[:2]
        self._frame_size = (w, h)

        if main_warmup_frames is not None:
            self.reset_tracking()

        inference = self._ensure_throw_inference()
        if inference is None:
            self._main_motion.build_mask(main_frame, main_previous_frame)
            self._secondary_motion.build_mask(
                secondary_frame, secondary_previous_frame
            )
            return main_frame.copy(), secondary_frame.copy()

        prediction = inference.predict(main_frame, warmup_frames=main_warmup_frames)
        throw_label = prediction.label

        main_motion_mask = self._main_motion.build_mask(
            main_frame,
            main_previous_frame,
            mog2_warmup_frames=main_mog2_warmup_frames,
        )
        secondary_motion_mask = self._secondary_motion.build_mask(
            secondary_frame,
            secondary_previous_frame,
            mog2_warmup_frames=secondary_mog2_warmup_frames,
        )

        main_result = self._main_tracker.update(
            throw_label=throw_label,
            wrist_pos=_extract_wrist_pos(prediction.detection),
            motion_mask=main_motion_mask,
            defer_detecting_throw=True,
        )
        secondary_result = self._secondary_tracker.update_secondary(
            throw_label=throw_label,
            motion_mask=secondary_motion_mask,
            defer_detecting_throw=True,
        )
        self._sync_stereo_phases()

        self._main_phase_prev = self._maybe_reset_pending_on_scan(
            main_result.phase, self._main_phase_prev, "main"
        )
        self._secondary_phase_prev = self._maybe_reset_pending_on_scan(
            secondary_result.phase, self._secondary_phase_prev, "secondary"
        )
        self._main_phase_prev = main_result.phase
        self._secondary_phase_prev = secondary_result.phase

        self._capture_detection(
            main_result.phase,
            main_result.detected_ball_pos,
            frame_index,
            self._main_pending,
        )
        self._capture_detection(
            secondary_result.phase,
            secondary_result.detected_ball_pos,
            frame_index,
            self._secondary_pending,
        )

        self._try_pair_throw()

        return main_frame.copy(), secondary_frame.copy()
