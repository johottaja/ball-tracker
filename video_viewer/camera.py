from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from .config import MAX_CAMERA_PROBE, TARGET_RECORD_FPS

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        if self.name == f"Camera {self.index}":
            return self.name
        return f"{self.name} ({self.index})"


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
    return TARGET_RECORD_FPS


def _darwin_camera_names() -> dict[int, str]:
    """Map OpenCV indices to device names (matches AVFoundation unique-id sort order)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        cameras = json.loads(result.stdout).get("SPCameraDataType", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return {}
    sorted_cameras = sorted(
        cameras,
        key=lambda camera: camera.get("spcamera_unique-id", ""),
    )
    return {index: camera["_name"] for index, camera in enumerate(sorted_cameras)}


def _linux_camera_names() -> dict[int, str]:
    v4l_root = Path("/sys/class/video4linux")
    if not v4l_root.is_dir():
        return {}
    names: dict[int, str] = {}
    for device in sorted(
        v4l_root.glob("video*"),
        key=lambda path: int(path.name.removeprefix("video")),
    ):
        suffix = device.name.removeprefix("video")
        if not suffix.isdigit():
            continue
        name_file = device / "name"
        if name_file.is_file():
            names[int(suffix)] = name_file.read_text().strip()
    return names


def get_camera_names() -> dict[int, str]:
    if sys.platform == "darwin":
        return _darwin_camera_names()
    if sys.platform.startswith("linux"):
        return _linux_camera_names()
    return {}


def camera_device(index: int, names: dict[int, str] | None = None) -> CameraDevice:
    if names is None:
        names = get_camera_names()
    return CameraDevice(index, names.get(index, f"Camera {index}"))


def probe_cameras(max_index: int = MAX_CAMERA_PROBE) -> list[CameraDevice]:
    """Return cameras that open and deliver at least one frame."""
    names = get_camera_names()
    available: list[CameraDevice] = []
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
            available.append(camera_device(index, names))
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
        # Read geometry before the capture thread starts; AVFoundation can crash
        # if the main thread touches the same VideoCapture concurrently.
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._consumer = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
            self._frame_id = 0

    def set_frame_consumer(self, consumer: FrameConsumer | None) -> None:
        with self._lock:
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
            with self._lock:
                consumer = self._consumer

            if consumer is not None:
                consumer(frame_copy)

            with self._lock:
                self._latest_frame = frame_copy
                self._frame_id += 1
