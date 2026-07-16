from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SyncEvent:
    frame_index: int
    sync_id: int
    offset: float

from pose_detection import DominantHandDetection, PoseDetector, detect_dominant_hand_detection
from pose_detection.types import PlayerSide
from throw_detection.inference import ThrowPrediction

from .ball_motion import BallDetectionMethod


@dataclass
class StreamPlaybackCache:
    """Per-stream playback caches for pose, GRU, motion masks, and filter outputs."""

    _pose: dict[int, DominantHandDetection | None] = field(default_factory=dict)
    _gru: dict[int, ThrowPrediction] = field(default_factory=dict)
    _player_pose: dict[tuple[PlayerSide, int], DominantHandDetection | None] = field(
        default_factory=dict
    )
    _player_gru: dict[tuple[PlayerSide, int], ThrowPrediction] = field(
        default_factory=dict
    )
    _motion_masks: dict[tuple[BallDetectionMethod, int], np.ndarray] = field(
        default_factory=dict
    )
    _filter_outputs: dict[tuple[str, str, int], np.ndarray] = field(
        default_factory=dict
    )

    def clear(self) -> None:
        self._pose.clear()
        self._gru.clear()
        self._player_pose.clear()
        self._player_gru.clear()
        self._motion_masks.clear()
        self._filter_outputs.clear()

    def clear_filter_outputs(self) -> None:
        self._filter_outputs.clear()

    def clear_motion_masks(self) -> None:
        self._motion_masks.clear()

    def has_pose(self, frame_index: int) -> bool:
        return frame_index in self._pose

    def get_pose(self, frame_index: int) -> DominantHandDetection | None:
        return self._pose[frame_index]

    def put_pose(self, frame_index: int, detection: DominantHandDetection | None) -> None:
        self._pose[frame_index] = detection

    def has_player_pose(self, player_side: PlayerSide, frame_index: int) -> bool:
        return (player_side, frame_index) in self._player_pose

    def get_player_pose(
        self, player_side: PlayerSide, frame_index: int
    ) -> DominantHandDetection | None:
        return self._player_pose[(player_side, frame_index)]

    def put_player_pose(
        self,
        player_side: PlayerSide,
        frame_index: int,
        detection: DominantHandDetection | None,
    ) -> None:
        self._player_pose[(player_side, frame_index)] = detection

    def has_gru(self, frame_index: int) -> bool:
        return frame_index in self._gru

    def get_gru(self, frame_index: int) -> ThrowPrediction:
        return self._gru[frame_index]

    def put_gru(self, frame_index: int, prediction: ThrowPrediction) -> None:
        self._gru[frame_index] = prediction

    def has_player_gru(self, player_side: PlayerSide, frame_index: int) -> bool:
        return (player_side, frame_index) in self._player_gru

    def get_player_gru(self, player_side: PlayerSide, frame_index: int) -> ThrowPrediction:
        return self._player_gru[(player_side, frame_index)]

    def put_player_gru(
        self,
        player_side: PlayerSide,
        frame_index: int,
        prediction: ThrowPrediction,
    ) -> None:
        self._player_gru[(player_side, frame_index)] = prediction

    def has_motion_mask(self, method: BallDetectionMethod, frame_index: int) -> bool:
        return (method, frame_index) in self._motion_masks

    def get_motion_mask(self, method: BallDetectionMethod, frame_index: int) -> np.ndarray:
        return self._motion_masks[(method, frame_index)]

    def put_motion_mask(
        self,
        method: BallDetectionMethod,
        frame_index: int,
        mask: np.ndarray,
    ) -> None:
        self._motion_masks[(method, frame_index)] = mask.copy()

    def has_filter_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
    ) -> bool:
        return (filter_id, ball_method.value, frame_index) in self._filter_outputs

    def get_filter_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
    ) -> np.ndarray:
        return self._filter_outputs[(filter_id, ball_method.value, frame_index)]

    def put_filter_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
        frame: np.ndarray,
    ) -> None:
        self._filter_outputs[(filter_id, ball_method.value, frame_index)] = frame.copy()


@dataclass
class PlaybackCache:
    """Playback caches for mono (main only) or stereo (main + secondary) streams."""

    main: StreamPlaybackCache = field(default_factory=StreamPlaybackCache)
    secondary: StreamPlaybackCache = field(default_factory=StreamPlaybackCache)
    _stereo_outputs: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    _sync_events: list[SyncEvent] = field(default_factory=list)

    def clear(self) -> None:
        self.main.clear()
        self.secondary.clear()
        self._stereo_outputs.clear()
        self._sync_events.clear()

    def clear_filter_outputs(self) -> None:
        self.main.clear_filter_outputs()
        self.secondary.clear_filter_outputs()
        self._stereo_outputs.clear()

    def clear_motion_masks(self) -> None:
        self.main.clear_motion_masks()
        self.secondary.clear_motion_masks()

    def put_sync_event(self, frame_index: int, sync_id: int, offset: float) -> None:
        self._sync_events = [
            event for event in self._sync_events if event.sync_id != sync_id
        ]
        self._sync_events.append(
            SyncEvent(frame_index=frame_index, sync_id=sync_id, offset=offset)
        )

    def latest_sync_at_or_before(self, frame_index: int) -> SyncEvent | None:
        candidates = [
            event for event in self._sync_events if event.frame_index <= frame_index
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda event: event.frame_index)

    def has_stereo_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
    ) -> bool:
        return (filter_id, ball_method.value, frame_index) in self._stereo_outputs

    def get_stereo_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._stereo_outputs[(filter_id, ball_method.value, frame_index)]

    def put_stereo_output(
        self,
        filter_id: str,
        ball_method: BallDetectionMethod,
        frame_index: int,
        main_frame: np.ndarray,
        secondary_frame: np.ndarray,
    ) -> None:
        self._stereo_outputs[(filter_id, ball_method.value, frame_index)] = (
            main_frame.copy(),
            secondary_frame.copy(),
        )


def cached_pose_detection(
    frame: np.ndarray,
    *,
    detector: PoseDetector | None = None,
    cache: StreamPlaybackCache | None = None,
    frame_index: int | None = None,
) -> DominantHandDetection | None:
    if cache is not None and frame_index is not None and cache.has_pose(frame_index):
        return cache.get_pose(frame_index)

    detection = detect_dominant_hand_detection(frame, detector=detector)
    if cache is not None and frame_index is not None:
        cache.put_pose(frame_index, detection)
    return detection
