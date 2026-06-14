from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2

from .config import MAX_CAMERA_PROBE, TARGET_RECORD_FPS

if TYPE_CHECKING:
    import numpy as np


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
    print(f"Reported FPS: {reported}")
    if reported is None or reported <= 1:
        print("Using default FPS")
        return TARGET_RECORD_FPS
    print(f"Using reported FPS: {reported}")
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


FrameConsumer = Callable[["np.ndarray"], None]


class CameraReader:
    """Capture frames on a background thread; expose the latest frame to the UI."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_id = 0
        self._consumer: FrameConsumer | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
            self._frame_id = 0
        self._consumer = None

    def set_frame_consumer(self, consumer: FrameConsumer | None) -> None:
        self._consumer = consumer

    def get_latest_frame(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0
            return True, self._latest_frame, self._frame_id

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_copy = frame.copy()
            consumer = self._consumer
            if consumer is not None:
                consumer(frame_copy)

            with self._lock:
                self._latest_frame = frame_copy
                self._frame_id += 1
