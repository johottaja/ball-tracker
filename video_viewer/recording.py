from __future__ import annotations

from pathlib import Path

import cv2

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

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return source_count

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
        return 0

    actual_source_count = len(frames)
    if actual_source_count >= target_count:
        return actual_source_count

    indices = source_indices_for_even_extension(actual_source_count, target_count)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    writer = create_writer(fps, width, height, temp_path)
    for src_index in indices:
        writer.write(frames[src_index])
    writer.release()

    if not temp_path.is_file():
        return actual_source_count

    path.unlink()
    temp_path.rename(path)
    return target_count
