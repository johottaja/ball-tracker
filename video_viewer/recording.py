from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_VIDEO


def create_writer(
    fps: float,
    width: int,
    height: int,
    path: Path = DEFAULT_VIDEO,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def source_indices_for_even_extension(
    source_count: int, target_count: int
) -> list[int]:
    """Map output frames to source frames, duplicating evenly to reach target_count."""
    if source_count <= 0:
        return []
    if target_count <= source_count:
        return list(range(target_count))
    if target_count == 1:
        return [0]
    return [
        round(i * (source_count - 1) / (target_count - 1))
        for i in range(target_count)
    ]


def indices_for_lagging_stream(
    lagging_timestamps: list[float],
    reference_timestamps: list[float],
) -> list[int]:
    """Map the lagging clip onto the reference clip's capture-time slots.

    At each reference timestamp the lagging stream keeps its latest frame until
    a new capture arrives, inserting a duplicate whenever it is one frame behind.
    """
    if not lagging_timestamps or not reference_timestamps:
        return []

    indices: list[int] = []
    source_index = 0
    for slot_time in reference_timestamps:
        while (
            source_index + 1 < len(lagging_timestamps)
            and lagging_timestamps[source_index + 1] <= slot_time
        ):
            source_index += 1
        indices.append(source_index)
    return indices


def _has_consecutive_duplicates(indices: list[int]) -> bool:
    return any(
        indices[index] == indices[index - 1] for index in range(1, len(indices))
    )


def _write_mp4_with_ffmpeg(
    path: Path,
    *,
    frames: list[np.ndarray],
    indices: list[int],
    fps: float,
    width: int,
    height: int,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.stdin is None:
        return False

    try:
        for source_index in indices:
            frame = frames[source_index]
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            proc.stdin.write(frame.tobytes())
    except (BrokenPipeError, OSError, ValueError):
        proc.stdin.close()
        proc.wait()
        return False

    proc.stdin.close()
    if proc.wait() != 0:
        return False
    return path.is_file() and path.stat().st_size > 0


def _reencode_video_with_indices(
    path: Path,
    *,
    indices: list[int],
    source_count: int,
    fps: float,
) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list = []
    for index in range(source_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        return False

    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if _has_consecutive_duplicates(indices) and _write_mp4_with_ffmpeg(
        temp_path,
        frames=frames,
        indices=indices,
        fps=fps,
        width=width,
        height=height,
    ):
        path.unlink(missing_ok=True)
        temp_path.rename(path)
        return True

    writer = create_writer(fps, width, height, temp_path)
    for source_index in indices:
        writer.write(frames[source_index])
    writer.release()

    if not temp_path.is_file():
        return False

    path.unlink(missing_ok=True)
    temp_path.rename(path)
    return True


def extend_video_evenly(
    path: Path,
    *,
    source_count: int,
    target_count: int,
    fps: float,
) -> int:
    """Re-encode a clip, duplicating frames evenly until target_count is reached."""
    if source_count <= 0 or source_count >= target_count:
        return source_count

    indices = source_indices_for_even_extension(source_count, target_count)
    if not _reencode_video_with_indices(
        path, indices=indices, source_count=source_count, fps=fps
    ):
        return source_count
    return target_count


def extend_video_to_reference(
    path: Path,
    *,
    source_timestamps: list[float],
    reference_timestamps: list[float],
    source_count: int,
    fps: float,
) -> int:
    """Re-encode the lagging clip to match the reference clip's frame count."""
    target_count = len(reference_timestamps)
    if source_count <= 0 or source_count >= target_count:
        return source_count

    indices = indices_for_lagging_stream(source_timestamps, reference_timestamps)
    if len(indices) != target_count:
        return source_count

    if not _reencode_video_with_indices(
        path, indices=indices, source_count=source_count, fps=fps
    ):
        return source_count
    return target_count
