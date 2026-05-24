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
