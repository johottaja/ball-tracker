from __future__ import annotations

from .config import (
    SYNC_COOLDOWN_SECONDS,
    SYNC_PAIRING_WINDOW_FRAMES,
    SYNC_TIMEOUT_FRAMES,
)
from .subframe import estimate_bounce_subframe_index
from .tracker import CameraSyncTracker
from .types import CameraSyncResult, FrameSyncResult, Phase


class FrameSyncEngine:
    """Coordinates two per-camera trackers and computes stereo sync offset."""

    def __init__(self) -> None:
        self._main = CameraSyncTracker()
        self._secondary = CameraSyncTracker()
        self._session_active = False
        self._session_start_frame: int | None = None
        self._session_start_time_s: float | None = None
        self._main_joined = False
        self._secondary_joined = False
        self._latest_offset: float | None = None
        self._sync_id = 0
        self._cooldown_until_frame: int | None = None

    def reset(self) -> None:
        self._main.reset()
        self._secondary.reset()
        self._session_active = False
        self._session_start_frame = None
        self._session_start_time_s = None
        self._main_joined = False
        self._secondary_joined = False
        self._cooldown_until_frame = None

    def restore_persisted_sync(self, *, offset: float, sync_id: int) -> None:
        """Restore offset measured earlier in playback (after a seek reset)."""
        self._latest_offset = offset
        self._sync_id = sync_id

    @property
    def latest_offset(self) -> float | None:
        return self._latest_offset

    @property
    def sync_id(self) -> int:
        return self._sync_id

    def update(
        self,
        frame_index: int,
        main_ball_bottom: tuple[int, int] | None,
        secondary_ball_bottom: tuple[int, int] | None,
        *,
        video_fps: float | None = None,
        main_native_frame_index: int | None = None,
        secondary_native_frame_index: int | None = None,
        main_capture_time_s: float | None = None,
        secondary_capture_time_s: float | None = None,
        main_fresh: bool = True,
        secondary_fresh: bool = True,
    ) -> FrameSyncResult:
        if self._in_cooldown(frame_index):
            return self._cooldown_result(main_ball_bottom, secondary_ball_bottom)

        prev_main_phase = self._main.phase
        prev_secondary_phase = self._secondary.phase

        main_result = self._main.update(
            frame_index, main_ball_bottom, native_frame_index=main_native_frame_index,
            capture_time_s=main_capture_time_s, fresh=main_fresh,
        )
        secondary_result = self._secondary.update(
            frame_index, secondary_ball_bottom, native_frame_index=secondary_native_frame_index,
            capture_time_s=secondary_capture_time_s, fresh=secondary_fresh,
        )

        if self._main.phase == Phase.SYNCING and prev_main_phase == Phase.WATCHING:
            self._on_camera_entered_syncing(
                is_main=True, frame_index=frame_index, time_s=main_capture_time_s
            )
        if self._secondary.phase == Phase.SYNCING and prev_secondary_phase == Phase.WATCHING:
            self._on_camera_entered_syncing(
                is_main=False, frame_index=frame_index, time_s=secondary_capture_time_s
            )

        if self._session_active:
            self._check_session_timeouts(
                frame_index, max(value for value in (main_capture_time_s, secondary_capture_time_s) if value is not None)
                if main_capture_time_s is not None or secondary_capture_time_s is not None else None,
                video_fps,
            )
            if self._both_done():
                self._complete_session(frame_index, video_fps)

        main_display, secondary_display = self._display_values()
        return FrameSyncResult(
            main=main_result,
            secondary=secondary_result,
            main_sync_display=main_display,
            secondary_sync_display=secondary_display,
            sync_id=self._sync_id,
        )

    def _display_values(self) -> tuple[float | None, float | None]:
        if self._latest_offset is None:
            return None, None
        return self._latest_offset, -self._latest_offset

    def _in_cooldown(self, frame_index: int) -> bool:
        return (
            self._cooldown_until_frame is not None
            and frame_index < self._cooldown_until_frame
        )

    def _cooldown_result(
        self,
        main_ball_bottom: tuple[int, int] | None,
        secondary_ball_bottom: tuple[int, int] | None,
    ) -> FrameSyncResult:
        main_display, secondary_display = self._display_values()
        return FrameSyncResult(
            main=CameraSyncResult(
                phase=Phase.WATCHING,
                detected_ball_bottom=main_ball_bottom,
            ),
            secondary=CameraSyncResult(
                phase=Phase.WATCHING,
                detected_ball_bottom=secondary_ball_bottom,
            ),
            main_sync_display=main_display,
            secondary_sync_display=secondary_display,
            sync_id=self._sync_id,
        )

    def _start_cooldown(self, frame_index: int, video_fps: float | None) -> None:
        fps = video_fps if video_fps and video_fps > 0 else 30.0
        cooldown_frames = max(1, round(fps * SYNC_COOLDOWN_SECONDS))
        self._cooldown_until_frame = frame_index + cooldown_frames

    def _on_camera_entered_syncing(
        self, *, is_main: bool, frame_index: int, time_s: float | None
    ) -> None:
        if not self._session_active:
            self._session_active = True
            self._session_start_frame = frame_index
            self._session_start_time_s = time_s
            self._main_joined = is_main
            self._secondary_joined = not is_main
            return

        if is_main:
            self._main_joined = True
        else:
            self._secondary_joined = True

    def _both_done(self) -> bool:
        return (
            self._session_active
            and self._main_joined
            and self._secondary_joined
            and self._main.phase == Phase.DONE
            and self._secondary.phase == Phase.DONE
        )

    def _check_session_timeouts(
        self, frame_index: int, time_s: float | None, video_fps: float | None
    ) -> None:
        if self._session_start_frame is None:
            return

        fps = video_fps if video_fps and video_fps > 0 else 30.0
        elapsed = (
            time_s - self._session_start_time_s
            if time_s is not None and self._session_start_time_s is not None
            else (frame_index - self._session_start_frame) / fps
        )

        if (
            elapsed > SYNC_PAIRING_WINDOW_FRAMES / fps
            and not (self._main_joined and self._secondary_joined)
        ):
            self._abort_session()
            return

        if elapsed > SYNC_TIMEOUT_FRAMES / fps and not self._both_done():
            self._abort_session()

    def _complete_session(
        self,
        frame_index: int,
        video_fps: float | None,
    ) -> None:
        main_bounce = self._main.bounce_interval
        secondary_bounce = self._secondary.bounce_interval
        if main_bounce is None or secondary_bounce is None:
            self._abort_session()
            return

        main_t = estimate_bounce_subframe_index(self._main.samples, main_bounce)
        secondary_t = estimate_bounce_subframe_index(
            self._secondary.samples,
            secondary_bounce,
        )
        if main_t is None or secondary_t is None:
            self._abort_session()
            return

        fps = video_fps if video_fps and video_fps > 0 else 30.0
        self._latest_offset = round((secondary_t - main_t) * fps, 2)
        self._sync_id += 1
        self._start_cooldown(frame_index, video_fps)
        self._end_session()

    def _abort_session(self) -> None:
        self._end_session()

    def _end_session(self) -> None:
        self._session_active = False
        self._session_start_frame = None
        self._session_start_time_s = None
        self._main_joined = False
        self._secondary_joined = False
        self._main.reset()
        self._secondary.reset()
