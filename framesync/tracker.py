from __future__ import annotations

from .config import (
    DROP_STREAK_FRAMES,
    MAX_HORIZONTAL_DELTA_PX,
    MIN_DOWNWARD_VY,
    POST_BOUNCE_CAPTURE_FRAMES,
    SLOWDOWN_RATIO,
)
from .types import BallSample, BounceInterval, CameraSyncResult, Phase


class CameraSyncTracker:
    """Per-camera state machine for straight-down drop and bounce capture."""

    def __init__(self) -> None:
        self.phase = Phase.WATCHING
        self._drop_streak = 0
        self._samples: list[BallSample] = []
        self._bounce_interval: BounceInterval | None = None
        self._capture_remaining = 0
        self._last_bottom: tuple[int, int] | None = None
        self._entered_syncing_frame: int | None = None

    def reset(self) -> None:
        self.phase = Phase.WATCHING
        self._drop_streak = 0
        self._samples = []
        self._bounce_interval = None
        self._capture_remaining = 0
        self._last_bottom = None
        self._entered_syncing_frame = None

    @property
    def bounce_interval(self) -> BounceInterval | None:
        return self._bounce_interval

    @property
    def samples(self) -> list[BallSample]:
        return list(self._samples)

    @property
    def entered_syncing_frame(self) -> int | None:
        return self._entered_syncing_frame

    def update(
        self,
        frame_index: int,
        ball_bottom: tuple[int, int] | None,
        *,
        native_frame_index: int | None = None,
        capture_time_s: float | None = None,
        fresh: bool = True,
    ) -> CameraSyncResult:
        detected = ball_bottom
        if not fresh:
            return CameraSyncResult(
                phase=self.phase, detected_ball_bottom=detected, samples=list(self._samples)
            )

        if self.phase == Phase.WATCHING:
            self._update_watching(frame_index, ball_bottom, native_frame_index, capture_time_s)
        elif self.phase == Phase.SYNCING:
            self._update_syncing(frame_index, ball_bottom, native_frame_index, capture_time_s)
        elif self.phase == Phase.CAPTURING:
            self._update_capturing(frame_index, ball_bottom, native_frame_index, capture_time_s)
        # DONE: hold state until engine resets after session completes.

        return CameraSyncResult(
            phase=self.phase,
            detected_ball_bottom=detected,
            samples=list(self._samples),
        )

    def _record_sample(
        self,
        frame_index: int,
        ball_bottom: tuple[int, int],
        native_frame_index: int | None,
        capture_time_s: float | None,
    ) -> None:
        sample = BallSample(
            frame_index=frame_index,
            native_frame_index=native_frame_index if native_frame_index is not None else frame_index,
            capture_time_s=capture_time_s if capture_time_s is not None else float(frame_index),
            bottom_x=ball_bottom[0],
            bottom_y=ball_bottom[1],
        )
        if self._samples and self._samples[-1].frame_index == frame_index:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        self._last_bottom = ball_bottom

    def _update_watching(
        self,
        frame_index: int,
        ball_bottom: tuple[int, int] | None,
        native_frame_index: int | None,
        capture_time_s: float | None,
    ) -> None:
        if ball_bottom is None or self._last_bottom is None:
            self._drop_streak = 0
            if ball_bottom is not None:
                self._last_bottom = ball_bottom
            return

        dx = abs(ball_bottom[0] - self._last_bottom[0])
        dy = ball_bottom[1] - self._last_bottom[1]
        if dy > 0 and dx <= MAX_HORIZONTAL_DELTA_PX:
            self._drop_streak += 1
        else:
            self._drop_streak = 0

        self._last_bottom = ball_bottom

        if self._drop_streak >= DROP_STREAK_FRAMES:
            self._enter_syncing(frame_index, ball_bottom, native_frame_index, capture_time_s)

    def _enter_syncing(
        self, frame_index: int, ball_bottom: tuple[int, int],
        native_frame_index: int | None, capture_time_s: float | None,
    ) -> None:
        self.phase = Phase.SYNCING
        self._entered_syncing_frame = frame_index
        self._samples = []
        self._record_sample(frame_index, ball_bottom, native_frame_index, capture_time_s)

    def _update_syncing(
        self,
        frame_index: int,
        ball_bottom: tuple[int, int] | None,
        native_frame_index: int | None,
        capture_time_s: float | None,
    ) -> None:
        if ball_bottom is not None:
            self._record_sample(frame_index, ball_bottom, native_frame_index, capture_time_s)
            if self._detect_bounce():
                self._enter_capturing(frame_index, ball_bottom, native_frame_index, capture_time_s)
        elif self._last_bottom is not None:
            self._last_bottom = None

    def _update_capturing(
        self,
        frame_index: int,
        ball_bottom: tuple[int, int] | None,
        native_frame_index: int | None,
        capture_time_s: float | None,
    ) -> None:
        if ball_bottom is not None:
            self._record_sample(frame_index, ball_bottom, native_frame_index, capture_time_s)
        self._capture_remaining -= 1
        if self._capture_remaining <= 0:
            self.phase = Phase.DONE

    def _enter_capturing(
        self, frame_index: int, ball_bottom: tuple[int, int],
        native_frame_index: int | None, capture_time_s: float | None,
    ) -> None:
        self.phase = Phase.CAPTURING
        self._capture_remaining = POST_BOUNCE_CAPTURE_FRAMES
        self._record_sample(frame_index, ball_bottom, native_frame_index, capture_time_s)

    def _detect_bounce(self) -> bool:
        if len(self._samples) < 3:
            return False

        prev_prev = self._samples[-3]
        prev_sample = self._samples[-2]
        curr_sample = self._samples[-1]

        vy_prev = prev_sample.bottom_y - prev_prev.bottom_y
        vy_curr = curr_sample.bottom_y - prev_sample.bottom_y

        sign_change = vy_prev > 0 and vy_curr < 0
        slowdown = (
            vy_prev > MIN_DOWNWARD_VY
            and vy_curr < vy_prev * SLOWDOWN_RATIO
        )
        if not sign_change and not slowdown:
            return False

        self._bounce_interval = BounceInterval(
            frame_prev=prev_sample.frame_index,
            frame_curr=curr_sample.frame_index,
        )
        return True
