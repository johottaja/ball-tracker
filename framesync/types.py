from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    WATCHING = "watching"
    SYNCING = "syncing"
    CAPTURING = "capturing"
    DONE = "done"


@dataclass(frozen=True)
class BallSample:
    frame_index: int
    native_frame_index: int
    capture_time_s: float
    bottom_x: int
    bottom_y: int


@dataclass
class BounceInterval:
    """Macro bounce between two consecutive frame indices."""

    frame_prev: int
    frame_curr: int


@dataclass
class CameraSyncResult:
    phase: Phase
    detected_ball_bottom: tuple[int, int] | None
    samples: list[BallSample] = field(default_factory=list)


@dataclass
class FrameSyncResult:
    main: CameraSyncResult
    secondary: CameraSyncResult
    main_sync_display: float | None
    secondary_sync_display: float | None
    sync_id: int
