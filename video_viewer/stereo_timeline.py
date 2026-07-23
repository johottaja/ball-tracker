from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

STEREO_TIMELINE_FILENAME = "stereo_timeline.json"
TIMELINE_VERSION = 2
Side = Literal["left", "right"]


@dataclass(frozen=True)
class StereoTimelineMetadata:
    version: int
    fps: float
    left_frame_count: int
    right_frame_count: int
    left_timestamps: list[float]
    right_timestamps: list[float]
    alignment: str = "bidirectional_pairing"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "fps": self.fps,
            "left_frame_count": self.left_frame_count,
            "right_frame_count": self.right_frame_count,
            "left_timestamps": self.left_timestamps,
            "right_timestamps": self.right_timestamps,
            "alignment": self.alignment,
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
            alignment=str(data.get("alignment", "legacy_reference")),
        )


@dataclass(frozen=True)
class StereoTimeline:
    """Immutable master slots that map each camera's native frames onto real time."""

    fps: float
    alignment: str
    captured_timestamps: bool
    master_times: tuple[float, ...]
    left_capture_times: tuple[float, ...]
    right_capture_times: tuple[float, ...]
    left_source_indices: tuple[int, ...]
    right_source_indices: tuple[int, ...]
    left_frame_count: int
    right_frame_count: int

    def __post_init__(self) -> None:
        slot_count = len(self.master_times)
        if slot_count == 0:
            raise ValueError("Stereo timeline has no common capture slots")
        if len(self.left_source_indices) != slot_count or len(self.right_source_indices) != slot_count:
            raise ValueError("Stereo timeline source maps do not match master slots")
        if len(self.left_capture_times) != self.left_frame_count:
            raise ValueError("Left capture timestamps do not match native frame count")
        if len(self.right_capture_times) != self.right_frame_count:
            raise ValueError("Right capture timestamps do not match native frame count")
        _validate_strictly_increasing(self.master_times, "master times")
        _validate_strictly_increasing(self.left_capture_times, "left capture times")
        _validate_strictly_increasing(self.right_capture_times, "right capture times")
        for side, indices, frame_count in (
            ("left", self.left_source_indices, self.left_frame_count),
            ("right", self.right_source_indices, self.right_frame_count),
        ):
            if any(index < 0 or index >= frame_count for index in indices):
                raise ValueError(f"{side} source map contains an out-of-range native index")
            if any(current < previous for previous, current in zip(indices, indices[1:])):
                raise ValueError(f"{side} source map goes backwards")
        if any(
            left == previous_left and right == previous_right
            for (previous_left, previous_right), (left, right) in zip(
                zip(self.left_source_indices, self.right_source_indices),
                zip(self.left_source_indices[1:], self.right_source_indices[1:]),
            )
        ):
            raise ValueError("A master slot must advance at least one camera")

    @property
    def master_count(self) -> int:
        return len(self.master_times)

    @property
    def signature(self) -> str:
        return _timeline_signature(self)

    def left_source_index(self, master_index: int) -> int:
        return self.left_source_indices[master_index]

    def right_source_index(self, master_index: int) -> int:
        return self.right_source_indices[master_index]

    def source_index(self, side: Side, master_index: int) -> int:
        if side == "left":
            return self.left_source_index(master_index)
        return self.right_source_index(master_index)

    def native_frame_count(self, side: Side) -> int:
        if side == "left":
            return self.left_frame_count
        return self.right_frame_count

    def is_hold(self, side: Side, master_index: int) -> bool:
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

    def capture_time(self, side: Side, master_index: int) -> float:
        """Actual capture time for the native frame shown in a master slot."""
        source_index = self.source_index(side, master_index)
        times = self.left_capture_times if side == "left" else self.right_capture_times
        return times[source_index]

    def duration_between_frames_s(self, start_frame: int, end_frame: int) -> float:
        if end_frame <= start_frame:
            return 0.0
        return self.master_times[end_frame] - self.master_times[start_frame]

    def native_neighbor_indices(
        self, side: Side, master_index: int
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
        _validate_capture_timestamps(left_timestamps, "left")
        _validate_capture_timestamps(right_timestamps, "right")
        left_count = len(left_timestamps)
        right_count = len(right_timestamps)
        overlap_start = max(left_timestamps[0], right_timestamps[0])
        overlap_end = min(left_timestamps[-1], right_timestamps[-1])
        if overlap_end <= overlap_start:
            raise ValueError("Camera recordings have no overlapping capture interval")
        left_indices, right_indices, absolute_master_times = _paired_slots(
            left_timestamps,
            right_timestamps,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
        )
        if not absolute_master_times:
            raise ValueError("No timestamp pairs could be formed in the common interval")
        origin = overlap_start
        master_times = _relative_timestamps(absolute_master_times, origin=origin)

        return cls(
            fps=fps,
            alignment="bidirectional_pairing",
            captured_timestamps=True,
            master_times=tuple(master_times),
            left_capture_times=tuple(_relative_timestamps(left_timestamps, origin=origin)),
            right_capture_times=tuple(_relative_timestamps(right_timestamps, origin=origin)),
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
        """Explicit synthetic fallback; never use it to produce 3D game output."""
        if left_frame_count <= 0 or right_frame_count <= 0:
            raise ValueError("Both cameras must contain at least one frame")
        master_count = min(left_frame_count, right_frame_count)
        frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0
        master_times = tuple(index * frame_time for index in range(master_count))
        left_capture_times = tuple(index * frame_time for index in range(left_frame_count))
        right_capture_times = tuple(index * frame_time for index in range(right_frame_count))
        left_indices = tuple(
            min(index, max(left_frame_count - 1, 0)) for index in range(master_count)
        )
        right_indices = tuple(
            min(index, max(right_frame_count - 1, 0)) for index in range(master_count)
        )
        return cls(
            fps=fps,
            alignment="synthetic_equal_index",
            captured_timestamps=False,
            master_times=master_times,
            left_capture_times=left_capture_times,
            right_capture_times=right_capture_times,
            left_source_indices=left_indices,
            right_source_indices=right_indices,
            left_frame_count=left_frame_count,
            right_frame_count=right_frame_count,
        )


def _relative_timestamps(timestamps: list[float], *, origin: float | None = None) -> list[float]:
    if origin is None:
        origin = timestamps[0]
    return [timestamp - origin for timestamp in timestamps]


def _validate_strictly_increasing(values: tuple[float, ...], label: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing")


def _validate_capture_timestamps(timestamps: list[float], side: str) -> None:
    if not timestamps:
        raise ValueError(f"{side} camera has no captured frames")
    _validate_strictly_increasing(tuple(timestamps), f"{side} capture timestamps")


def _median_period(timestamps: list[float]) -> float:
    deltas = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
    return statistics.median(deltas) if deltas else 1.0 / 30.0


def _paired_slots(
    left_times: list[float],
    right_times: list[float],
    *,
    overlap_start: float,
    overlap_end: float,
) -> tuple[list[int], list[int], list[float]]:
    """Greedily pair nearby captures; otherwise advance one camera and hold peer."""
    tolerance = min(max(_median_period(left_times), _median_period(right_times)) / 2.0, 0.100)
    left_pos = next((i for i, value in enumerate(left_times) if value >= overlap_start), len(left_times))
    right_pos = next((i for i, value in enumerate(right_times) if value >= overlap_start), len(right_times))
    if left_pos == len(left_times) or right_pos == len(right_times):
        return [], [], []

    # Drop leading unmatched frames. A slot must never show a future frame from
    # the other camera, and no source frame may precede the common interval.
    while (
        left_pos < len(left_times)
        and right_pos < len(right_times)
        and left_times[left_pos] <= overlap_end
        and right_times[right_pos] <= overlap_end
        and abs(left_times[left_pos] - right_times[right_pos]) > tolerance
    ):
        if left_times[left_pos] < right_times[right_pos]:
            left_pos += 1
        else:
            right_pos += 1

    if left_pos >= len(left_times) or right_pos >= len(right_times):
        return [], [], []
    if left_times[left_pos] > overlap_end or right_times[right_pos] > overlap_end:
        return [], [], []

    left_indices = [left_pos]
    right_indices = [right_pos]
    master_times = [max(left_times[left_pos], right_times[right_pos])]
    left_pos += 1
    right_pos += 1

    while True:
        left_time = left_times[left_pos] if left_pos < len(left_times) else math.inf
        right_time = right_times[right_pos] if right_pos < len(right_times) else math.inf
        if min(left_time, right_time) > overlap_end:
            break
        if left_time <= overlap_end and right_time <= overlap_end and abs(left_time - right_time) <= tolerance:
            next_left, next_right = left_pos, right_pos
            event_time = max(left_time, right_time)
            left_pos += 1
            right_pos += 1
        elif left_time < right_time:
            next_left, next_right = left_pos, right_indices[-1]
            event_time = left_time
            left_pos += 1
        else:
            next_left, next_right = left_indices[-1], right_pos
            event_time = right_time
            right_pos += 1
        if event_time <= master_times[-1]:
            raise ValueError("Capture events must yield strictly increasing master times")
        left_indices.append(next_left)
        right_indices.append(next_right)
        master_times.append(event_time)
    return left_indices, right_indices, master_times


def _timeline_signature(timeline: StereoTimeline) -> str:
    payload = {
        "alignment": timeline.alignment,
        "left_count": timeline.left_frame_count,
        "right_count": timeline.right_frame_count,
        "left_times": timeline.left_capture_times,
        "right_times": timeline.right_capture_times,
        "left_map": timeline.left_source_indices,
        "right_map": timeline.right_source_indices,
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def stereo_timeline_path_for(left_video: Path) -> Path:
    return left_video.parent / STEREO_TIMELINE_FILENAME


def save_stereo_timeline(path: Path, metadata: StereoTimelineMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_stereo_timeline(
    path: Path,
    *,
    left_frame_count: int | None = None,
    right_frame_count: int | None = None,
) -> StereoTimeline | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        metadata = StereoTimelineMetadata.from_dict(data)
        if left_frame_count is not None and metadata.left_frame_count != left_frame_count:
            return None
        if right_frame_count is not None and metadata.right_frame_count != right_frame_count:
            return None
        if len(metadata.left_timestamps) != metadata.left_frame_count:
            return None
        if len(metadata.right_timestamps) != metadata.right_frame_count:
            return None
        return StereoTimeline.from_metadata(metadata)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def load_stereo_timeline_for_videos(
    left_video: Path,
    *,
    left_frame_count: int,
    right_frame_count: int,
    fps: float,
) -> StereoTimeline:
    timeline = load_stereo_timeline(
        stereo_timeline_path_for(left_video),
        left_frame_count=left_frame_count,
        right_frame_count=right_frame_count,
    )
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
        version=TIMELINE_VERSION,
        fps=fps,
        left_frame_count=len(left_timestamps),
        right_frame_count=len(right_timestamps),
        left_timestamps=list(left_timestamps),
        right_timestamps=list(right_timestamps),
        alignment=timeline.alignment,
    )
    save_stereo_timeline(stereo_timeline_path_for(left_video), metadata)
    return timeline
