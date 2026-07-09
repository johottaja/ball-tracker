from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .recording import indices_for_lagging_stream

STEREO_TIMELINE_FILENAME = "stereo_timeline.json"


@dataclass(frozen=True)
class StereoTimelineMetadata:
    version: int
    fps: float
    left_frame_count: int
    right_frame_count: int
    left_timestamps: list[float]
    right_timestamps: list[float]
    reference: Literal["left", "right"]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "fps": self.fps,
            "left_frame_count": self.left_frame_count,
            "right_frame_count": self.right_frame_count,
            "left_timestamps": self.left_timestamps,
            "right_timestamps": self.right_timestamps,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StereoTimelineMetadata:
        return cls(
            version=int(data.get("version", 1)),
            fps=float(data.get("fps", 30.0)),
            left_frame_count=int(data["left_frame_count"]),
            right_frame_count=int(data["right_frame_count"]),
            left_timestamps=[float(value) for value in data["left_timestamps"]],
            right_timestamps=[float(value) for value in data["right_timestamps"]],
            reference=data.get("reference", "left"),
        )


@dataclass(frozen=True)
class StereoTimeline:
    """Master playback timeline with per-slot native source indices."""

    fps: float
    reference: Literal["left", "right"]
    master_times: tuple[float, ...]
    left_source_indices: tuple[int, ...]
    right_source_indices: tuple[int, ...]
    left_frame_count: int
    right_frame_count: int

    @property
    def master_count(self) -> int:
        return len(self.master_times)

    def left_source_index(self, master_index: int) -> int:
        return self.left_source_indices[master_index]

    def right_source_index(self, master_index: int) -> int:
        return self.right_source_indices[master_index]

    def source_index(self, side: Literal["left", "right"], master_index: int) -> int:
        if side == "left":
            return self.left_source_index(master_index)
        return self.right_source_index(master_index)

    def native_frame_count(self, side: Literal["left", "right"]) -> int:
        if side == "left":
            return self.left_frame_count
        return self.right_frame_count

    def is_hold(self, side: Literal["left", "right"], master_index: int) -> bool:
        if master_index <= 0:
            return False
        return self.source_index(side, master_index) == self.source_index(
            side, master_index - 1
        )

    def slot_duration_s(self, master_index: int) -> float:
        frame_time = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        if master_index + 1 >= self.master_count:
            return frame_time
        delta = self.master_times[master_index + 1] - self.master_times[master_index]
        return max(delta, frame_time / 4.0)

    def slot_duration_ms(self, master_index: int) -> int:
        return max(1, int(round(self.slot_duration_s(master_index) * 1000.0)))

    def master_time_s(self, master_index: int) -> float:
        return self.master_times[master_index]

    def time_at_frame(self, frame: int) -> float:
        return self.master_times[frame]

    def duration_between_frames_s(self, start_frame: int, end_frame: int) -> float:
        if end_frame <= start_frame:
            return 0.0
        return self.master_times[end_frame] - self.master_times[start_frame]

    def native_neighbor_indices(
        self, side: Literal["left", "right"], master_index: int
    ) -> tuple[int | None, int | None]:
        """Source indices before/after the current capture for frame differencing."""
        source_index = self.source_index(side, master_index)
        native_count = self.native_frame_count(side)
        previous_index = source_index - 1 if source_index > 0 else None
        next_index = source_index + 1 if source_index + 1 < native_count else None
        return previous_index, next_index

    @classmethod
    def from_capture(
        cls,
        *,
        left_timestamps: list[float],
        right_timestamps: list[float],
        fps: float,
    ) -> StereoTimeline:
        left_count = len(left_timestamps)
        right_count = len(right_timestamps)
        if left_count == 0 or right_count == 0:
            raise ValueError("Both cameras must have at least one captured frame")

        if left_count >= right_count:
            reference: Literal["left", "right"] = "left"
            master_times = _relative_timestamps(left_timestamps)
            left_indices = list(range(left_count))
            right_indices = indices_for_lagging_stream(right_timestamps, left_timestamps)
        else:
            reference = "right"
            master_times = _relative_timestamps(right_timestamps)
            right_indices = list(range(right_count))
            left_indices = indices_for_lagging_stream(left_timestamps, right_timestamps)

        if len(left_indices) != len(master_times) or len(right_indices) != len(master_times):
            raise ValueError("Stereo timeline index maps do not match master length")

        return cls(
            fps=fps,
            reference=reference,
            master_times=tuple(master_times),
            left_source_indices=tuple(left_indices),
            right_source_indices=tuple(right_indices),
            left_frame_count=left_count,
            right_frame_count=right_count,
        )

    @classmethod
    def from_metadata(cls, metadata: StereoTimelineMetadata) -> StereoTimeline:
        return cls.from_capture(
            left_timestamps=metadata.left_timestamps,
            right_timestamps=metadata.right_timestamps,
            fps=metadata.fps,
        )

    @classmethod
    def from_equal_index(
        cls,
        *,
        left_frame_count: int,
        right_frame_count: int,
        fps: float,
    ) -> StereoTimeline:
        """Fallback when no capture timestamps are available."""
        master_count = min(left_frame_count, right_frame_count)
        if master_count <= 0:
            master_count = max(left_frame_count, right_frame_count)
        frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0
        master_times = tuple(index * frame_time for index in range(master_count))
        left_indices = tuple(
            min(index, max(left_frame_count - 1, 0)) for index in range(master_count)
        )
        right_indices = tuple(
            min(index, max(right_frame_count - 1, 0)) for index in range(master_count)
        )
        reference: Literal["left", "right"] = (
            "left" if left_frame_count >= right_frame_count else "right"
        )
        return cls(
            fps=fps,
            reference=reference,
            master_times=master_times,
            left_source_indices=left_indices,
            right_source_indices=right_indices,
            left_frame_count=left_frame_count,
            right_frame_count=right_frame_count,
        )


def _relative_timestamps(timestamps: list[float]) -> list[float]:
    origin = timestamps[0]
    return [timestamp - origin for timestamp in timestamps]


def stereo_timeline_path_for(left_video: Path) -> Path:
    return left_video.parent / STEREO_TIMELINE_FILENAME


def save_stereo_timeline(path: Path, metadata: StereoTimelineMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_stereo_timeline(path: Path) -> StereoTimeline | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return StereoTimeline.from_metadata(StereoTimelineMetadata.from_dict(data))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def load_stereo_timeline_for_videos(
    left_video: Path,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> StereoTimeline:
    timeline = load_stereo_timeline(stereo_timeline_path_for(left_video))
    if timeline is not None:
        return timeline
    return StereoTimeline.from_equal_index(
        left_frame_count=left_frame_count,
        right_frame_count=right_frame_count,
        fps=fps,
    )


def finalize_stereo_recording(
    *,
    left_timestamps: list[float],
    right_timestamps: list[float],
    fps: float,
    left_video: Path,
) -> StereoTimeline:
    timeline = StereoTimeline.from_capture(
        left_timestamps=left_timestamps,
        right_timestamps=right_timestamps,
        fps=fps,
    )
    metadata = StereoTimelineMetadata(
        version=1,
        fps=fps,
        left_frame_count=len(left_timestamps),
        right_frame_count=len(right_timestamps),
        left_timestamps=list(left_timestamps),
        right_timestamps=list(right_timestamps),
        reference=timeline.reference,
    )
    save_stereo_timeline(stereo_timeline_path_for(left_video), metadata)
    return timeline
