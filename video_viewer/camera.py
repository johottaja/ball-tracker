from __future__ import annotations

import sys

import cv2

from .config import MAX_CAMERA_PROBE, TARGET_RECORD_FPS


def open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


def configure_camera_fps(cap: cv2.VideoCapture) -> float:
    """Request target fps from the device; ignore under-reported rates (common on macOS)."""
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, TARGET_RECORD_FPS)
    reported = cap.get(cv2.CAP_PROP_FPS)
    if reported is None or reported <= 1:
        return TARGET_RECORD_FPS
    return max(TARGET_RECORD_FPS, reported)


def probe_cameras(max_index: int = MAX_CAMERA_PROBE) -> list[int]:
    """Return indices of cameras that open and deliver at least one frame."""
    available: list[int] = []
    consecutive_failures = 0
    for index in range(max_index):
        cap = open_camera(index)
        if not cap.isOpened():
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            available.append(index)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
    return available
