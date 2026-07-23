from __future__ import annotations

import tkinter as tk
from collections import deque
from collections.abc import Callable
import time
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from video_viewer.display import fit_size

from .config import (
    FINGERTIP_FEED_INTERVAL_MS,
    FINGERTIP_MAX_DISTANCE_FROM_CORNER_PX,
    FINGERTIP_MAX_MISSING_SECONDS,
    FINGERTIP_MAX_STABILITY_RADIUS_PX,
    FINGERTIP_STABILITY_SECONDS,
    HAND_LANDMARKER_MODEL,
)
from .fingertip import FingertipDetection, FingertipDetector
from .homography import build_table_calibration, draw_table_xy_grid_on_image
from .refine import refine_calibration_from_fingertips
from .storage import load_calibration, save_calibration
from .types import TableCalibration

CORNER_COUNT = 4
# Bright blue, slightly dark (BGR).
MARKER_COLOR_BGR = (200, 120, 0)
MARKER_LINE_THICKNESS = 2
TABLE_GRID_CELL_M = 0.15
TABLE_GRID_COLOR_BGR = (239, 207, 158)  # #9ecfef
TABLE_GRID_THICKNESS = 1


class TableCalibrationDialog:
    """Table-corner clicking UI for stereo camera calibration (one feed at a time)."""

    def __init__(
        self,
        parent: tk.Tk,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        max_total_size: tuple[int, int],
        *,
        on_save: Callable[[TableCalibration], None] | None = None,
        frame_provider: Callable[[], tuple[np.ndarray, np.ndarray] | None] | None = None,
    ) -> None:
        self._parent = parent
        self._on_save = on_save
        self._frame_provider = frame_provider
        self._left_base = left_frame.copy()
        self._right_base = right_frame.copy()
        self._max_total_size = max_total_size
        self._display_size = fit_size(
            left_frame.shape[1],
            left_frame.shape[0],
            max_total_size,
        )
        self._active_camera = 0  # 0 = left, 1 = right
        self._left_corners: list[tuple[int, int]] = []
        self._right_corners: list[tuple[int, int]] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._window: tk.Toplevel | None = None
        self._image_label: ttk.Label | None = None
        self._save_btn: ttk.Button | None = None
        self._refresh_pending = False
        self._instruction_var: tk.StringVar | None = None
        self._finger_phase = False
        self._rough_calibration: TableCalibration | None = None
        self._fingertip_detector: FingertipDetector | None = None
        self._fingertip_corner_index = 0
        self._left_fingertips: list[tuple[float, float]] = []
        self._right_fingertips: list[tuple[float, float]] = []
        self._fingertip_samples: deque[
            tuple[float, FingertipDetection, FingertipDetection]
        ] = deque()
        self._last_fingertip_detection_s: float | None = None
        self._fingertip_feedback = "Waiting for fingertips in both feeds."

        existing = load_calibration()
        if existing is not None:
            self._default_length = str(existing.table_length_m)
            self._default_width = str(existing.table_width_m)
        else:
            self._default_length = "2.44"
            self._default_width = "0.61"

    def show(self) -> None:
        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        win = tk.Toplevel(self._parent)
        win.title("Table calibration")
        win.transient(self._parent)
        win.grab_set()
        self._window = win

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        win.geometry(f"{int(screen_w * 0.9)}x{int(screen_h * 0.9)}")
        win.minsize(self._max_total_size[0], self._max_total_size[1] + 220)

        root = ttk.Frame(win, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        footer = ttk.Frame(root)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        btn_row = ttk.Frame(footer)
        btn_row.pack(fill=tk.X, pady=(12, 0))

        self._save_btn = ttk.Button(
            btn_row,
            text="Start fingertip refinement",
            command=self._save,
            state=tk.DISABLED,
        )
        self._save_btn.pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Reset", command=self._reset_corners).pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 4))

        dims = ttk.Frame(footer)
        dims.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(dims, text="Table length (m):").pack(side=tk.LEFT)
        self._length_var = tk.StringVar(value=self._default_length)
        ttk.Entry(dims, textvariable=self._length_var, width=10).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(dims, text="Table width (m):").pack(side=tk.LEFT)
        self._width_var = tk.StringVar(value=self._default_width)
        ttk.Entry(dims, textvariable=self._width_var, width=10).pack(side=tk.LEFT, padx=(4, 0))
        self._length_var.trace_add("write", lambda *_: self._refresh_image())
        self._width_var.trace_add("write", lambda *_: self._refresh_image())

        header = ttk.Frame(root)
        header.pack(side=tk.TOP, fill=tk.X)

        self._instruction_var = tk.StringVar(
            value=(
                "Click four table corners on each feed in the same order (clockwise from "
                "above): 1 (+length,+width), 2 (+length,−width), 3 (−length,−width), "
                "4 (−length,+width). Origin is table center; length = +X, width = +Y. "
                "Click near an existing marker to adjust it. Use Reset to start over."
            )
        )
        ttk.Label(
            header,
            textvariable=self._instruction_var,
            wraplength=int(screen_w * 0.85),
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))

        camera_row = ttk.Frame(header)
        camera_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(camera_row, text="Camera:").pack(side=tk.LEFT)
        self._camera_var = tk.StringVar(value="left")
        ttk.Radiobutton(
            camera_row,
            text="Left",
            value="left",
            variable=self._camera_var,
            command=self._on_camera_changed,
        ).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Radiobutton(
            camera_row,
            text="Right",
            value="right",
            variable=self._camera_var,
            command=self._on_camera_changed,
        ).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value=self._status_text())
        ttk.Label(header, textvariable=self._status_var).pack(fill=tk.X, pady=(0, 8))

        self._image_label = ttk.Label(root, anchor=tk.CENTER)
        self._image_label.pack(fill=tk.BOTH, expand=True)
        self._image_label.bind("<Button-1>", self._on_click)
        self._image_label.bind("<Configure>", self._on_image_configure)

        win.protocol("WM_DELETE_WINDOW", self._close)
        self._refresh_image()
        self._update_save_state()

    def _corners_for_camera(self, camera: int) -> list[tuple[int, int]]:
        return self._left_corners if camera == 0 else self._right_corners

    def _camera_label(self, camera: int) -> str:
        return "Left" if camera == 0 else "Right"

    def _status_text(self) -> str:
        active = self._camera_label(self._active_camera)
        active_corners = self._corners_for_camera(self._active_camera)
        return (
            f"{active} camera: {len(active_corners)}/{CORNER_COUNT} corners   "
            f"(Left {len(self._left_corners)}/{CORNER_COUNT}, "
            f"Right {len(self._right_corners)}/{CORNER_COUNT})"
        )

    def _fit_to_area(self, frame_w: int, frame_h: int, area_w: int, area_h: int) -> tuple[int, int]:
        scale = min(area_w / frame_w, area_h / frame_h)
        return max(1, int(frame_w * scale)), max(1, int(frame_h * scale))

    def _current_display_size(self) -> tuple[int, int]:
        if self._image_label is None:
            return self._display_size
        label_w = max(1, self._image_label.winfo_width())
        label_h = max(1, self._image_label.winfo_height())
        base = self._left_base if self._active_camera == 0 else self._right_base
        return self._fit_to_area(base.shape[1], base.shape[0], label_w, label_h)

    def _image_offset(self) -> tuple[int, int]:
        if self._image_label is None:
            return 0, 0
        label_w = max(1, self._image_label.winfo_width())
        label_h = max(1, self._image_label.winfo_height())
        disp_w, disp_h = self._display_size
        return (label_w - disp_w) // 2, (label_h - disp_h) // 2

    def _on_camera_changed(self) -> None:
        self._active_camera = 0 if self._camera_var.get() == "left" else 1
        self._status_var.set(self._status_text())
        self._refresh_image()

    def _on_image_configure(self, event: tk.Event) -> None:
        if self._finger_phase:
            return
        if self._image_label is None:
            return
        new_size = self._current_display_size()
        if new_size == self._display_size:
            return
        self._display_size = new_size
        if self._refresh_pending:
            return
        self._refresh_pending = True
        self._image_label.after_idle(self._finish_resize_refresh)

    def _finish_resize_refresh(self) -> None:
        self._refresh_pending = False
        self._refresh_image()

    def _marker_radius(self, frame: np.ndarray) -> int:
        return max(6, min(frame.shape[0], frame.shape[1]) // 80)

    def _table_dimensions(self) -> tuple[float, float] | None:
        if self._length_var is None or self._width_var is None:
            return None
        try:
            length_m = float(self._length_var.get())
            width_m = float(self._width_var.get())
        except (ValueError, tk.TclError):
            return None
        if length_m <= 0 or width_m <= 0:
            return None
        return length_m, width_m

    def _draw_overlay(self, frame: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
        out = frame.copy()
        if not corners:
            return out

        dimensions = self._table_dimensions()
        if len(corners) == CORNER_COUNT and dimensions is not None:
            length_m, width_m = dimensions
            out = draw_table_xy_grid_on_image(
                out,
                [(float(x), float(y)) for x, y in corners],
                length_m=length_m,
                width_m=width_m,
                cell_size_m=TABLE_GRID_CELL_M,
                color_bgr=TABLE_GRID_COLOR_BGR,
                thickness=TABLE_GRID_THICKNESS,
            )

        radius = self._marker_radius(frame)
        color = MARKER_COLOR_BGR
        thickness = MARKER_LINE_THICKNESS

        for x, y in corners:
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(out, corners[i], corners[i + 1], color, thickness, lineType=cv2.LINE_AA)

        if len(corners) == CORNER_COUNT:
            cv2.line(out, corners[-1], corners[0], color, thickness, lineType=cv2.LINE_AA)
            cx = int(sum(p[0] for p in corners) / CORNER_COUNT)
            cy = int(sum(p[1] for p in corners) / CORNER_COUNT)
            cv2.circle(out, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

        return out

    def _frame_photo(self, frame: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(self._display_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _refresh_image(self) -> None:
        if self._image_label is None:
            return
        self._display_size = self._current_display_size()
        base = self._left_base if self._active_camera == 0 else self._right_base
        corners = self._left_corners if self._active_camera == 0 else self._right_corners
        frame = self._draw_overlay(base, corners)
        self._photo = self._frame_photo(frame)
        self._image_label.configure(image=self._photo)

    def _display_to_image_coords(self, display_x: int, display_y: int) -> tuple[int, int] | None:
        offset_x, offset_y = self._image_offset()
        local_x = display_x - offset_x
        local_y = display_y - offset_y
        disp_w, disp_h = self._display_size
        if local_x < 0 or local_y < 0 or local_x >= disp_w or local_y >= disp_h:
            return None

        base = self._left_base if self._active_camera == 0 else self._right_base
        frame_h, frame_w = base.shape[:2]
        x = int(local_x * frame_w / disp_w)
        y = int(local_y * frame_h / disp_h)
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        return x, y

    def _nearest_corner_index(
        self,
        point: tuple[int, int],
        corners: list[tuple[int, int]],
        frame: np.ndarray,
    ) -> int | None:
        if not corners:
            return None

        threshold = self._marker_radius(frame) * 2
        threshold_sq = threshold * threshold
        best_index: int | None = None
        best_dist_sq = threshold_sq
        px, py = point
        for index, (corner_x, corner_y) in enumerate(corners):
            dx = px - corner_x
            dy = py - corner_y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_index = index
        return best_index

    def _on_click(self, event: tk.Event) -> None:
        if self._finger_phase:
            return
        corners = self._left_corners if self._active_camera == 0 else self._right_corners
        base = self._left_base if self._active_camera == 0 else self._right_base

        point = self._display_to_image_coords(event.x, event.y)
        if point is None:
            return
        existing_index = self._nearest_corner_index(point, corners, base)
        if existing_index is not None:
            corners[existing_index] = point
        elif len(corners) < CORNER_COUNT:
            corners.append(point)
        else:
            return

        self._status_var.set(self._status_text())
        self._refresh_image()
        self._update_save_state()

    def _update_save_state(self) -> None:
        if self._finger_phase:
            return
        complete = (
            len(self._left_corners) == CORNER_COUNT
            and len(self._right_corners) == CORNER_COUNT
        )
        state = tk.NORMAL if complete else tk.DISABLED
        if self._save_btn is not None:
            self._save_btn.configure(state=state)

    def _reset_corners(self) -> None:
        if self._finger_phase:
            return
        self._left_corners.clear()
        self._right_corners.clear()
        self._active_camera = 0
        self._camera_var.set("left")
        self._status_var.set(self._status_text())
        self._refresh_image()
        self._update_save_state()

    def _save(self) -> None:
        if self._finger_phase:
            self._finish_fingertip_refinement()
            return
        try:
            length_m = float(self._length_var.get())
            width_m = float(self._width_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Table length and width must be numeric.",
                parent=self._window,
            )
            return

        if length_m <= 0 or width_m <= 0:
            messagebox.showerror(
                "Invalid input",
                "Table length and width must be positive.",
                parent=self._window,
            )
            return

        image_height, image_width = self._left_base.shape[:2]

        try:
            calibration = build_table_calibration(
                length_m=length_m,
                width_m=width_m,
                image_width=image_width,
                image_height=image_height,
                left_corners=[(float(x), float(y)) for x, y in self._left_corners],
                right_corners=[(float(x), float(y)) for x, y in self._right_corners],
            )
        except ValueError as exc:
            messagebox.showerror("Calibration failed", str(exc), parent=self._window)
            return

        if self._frame_provider is None:
            messagebox.showerror(
                "Live frames unavailable",
                "Fingertip refinement requires an active stereo camera feed.",
                parent=self._window,
            )
            return
        try:
            self._fingertip_detector = FingertipDetector(HAND_LANDMARKER_MODEL)
        except (FileNotFoundError, RuntimeError) as exc:
            messagebox.showerror("Fingertip detector unavailable", str(exc), parent=self._window)
            return

        self._rough_calibration = calibration
        self._finger_phase = True
        self._fingertip_corner_index = 0
        self._left_fingertips.clear()
        self._right_fingertips.clear()
        self._fingertip_samples.clear()
        self._last_fingertip_detection_s = None
        self._fingertip_feedback = "Place your fingertip on corner 1."
        if self._instruction_var is not None:
            self._instruction_var.set(
                "Fingertip refinement: place your index fingertip on the highlighted "
                "table corner in both feeds. Keep it still for 3 seconds. "
                "Corners are captured automatically."
            )
        if self._save_btn is not None:
            self._save_btn.configure(text="Save refined calibration", state=tk.DISABLED)
        self._refresh_fingertip_feed()

    def _expected_corner_pixels(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        index = self._fingertip_corner_index
        return self._left_corners[index], self._right_corners[index]

    @staticmethod
    def _stable_point(
        samples: list[FingertipDetection],
    ) -> tuple[float, float] | None:
        if not samples:
            return None
        points = np.array([(sample.x, sample.y) for sample in samples], dtype=np.float64)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        if float(np.max(distances)) > FINGERTIP_MAX_STABILITY_RADIUS_PX:
            return None
        return float(center[0]), float(center[1])

    @staticmethod
    def _stability_radius(samples: list[FingertipDetection]) -> float:
        if not samples:
            return float("inf")
        points = np.array([(sample.x, sample.y) for sample in samples], dtype=np.float64)
        center = np.median(points, axis=0)
        return float(np.max(np.linalg.norm(points - center, axis=1)))

    def _draw_fingertip_overlay(
        self,
        frame: np.ndarray,
        *,
        expected: tuple[float, float],
        detected: FingertipDetection | None,
        label: str,
        elapsed_s: float,
    ) -> np.ndarray:
        out = frame.copy()
        expected_point = (int(round(expected[0])), int(round(expected[1])))
        cv2.circle(out, expected_point, 22, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"Corner {self._fingertip_corner_index + 1}",
            (expected_point[0] + 26, expected_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if detected is not None:
            point = (int(round(detected.x)), int(round(detected.y)))
            cv2.circle(out, point, 9, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"{label}: hold still {min(elapsed_s, FINGERTIP_STABILITY_SECONDS):.1f}/"
            f"{FINGERTIP_STABILITY_SECONDS:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            self._fingertip_feedback,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def _show_stereo_frame(self, left: np.ndarray, right: np.ndarray) -> None:
        if self._image_label is None:
            return
        combined = np.hstack((left, right))
        label_w = max(1, self._image_label.winfo_width())
        label_h = max(1, self._image_label.winfo_height())
        display_size = self._fit_to_area(
            combined.shape[1],
            combined.shape[0],
            label_w,
            label_h,
        )
        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(display_size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self._image_label.configure(image=self._photo)

    def _refresh_fingertip_feed(self) -> None:
        if not self._finger_phase or self._window is None or not self._window.winfo_exists():
            return
        if self._frame_provider is None or self._fingertip_detector is None:
            return
        frames = self._frame_provider()
        if frames is None:
            if self._status_var is not None:
                self._status_var.set("Waiting for both live camera frames…")
            self._window.after(FINGERTIP_FEED_INTERVAL_MS, self._refresh_fingertip_feed)
            return

        left_frame, right_frame = frames
        expected_left, expected_right = self._expected_corner_pixels()
        left_tip = self._fingertip_detector.detect_nearest(
            left_frame,
            expected_left,
            max_distance_px=FINGERTIP_MAX_DISTANCE_FROM_CORNER_PX,
        )
        right_tip = self._fingertip_detector.detect_nearest(
            right_frame,
            expected_right,
            max_distance_px=FINGERTIP_MAX_DISTANCE_FROM_CORNER_PX,
        )
        now = time.monotonic()
        if left_tip is None or right_tip is None:
            if (
                self._last_fingertip_detection_s is not None
                and now - self._last_fingertip_detection_s > FINGERTIP_MAX_MISSING_SECONDS
            ):
                self._fingertip_samples.clear()
                self._last_fingertip_detection_s = None
        else:
            self._last_fingertip_detection_s = now
            self._fingertip_samples.append((now, left_tip, right_tip))

        elapsed_s = (
            now - self._fingertip_samples[0][0] if self._fingertip_samples else 0.0
        )
        left_overlay = self._draw_fingertip_overlay(
            left_frame,
            expected=expected_left,
            detected=left_tip,
            label="Left",
            elapsed_s=elapsed_s,
        )
        right_overlay = self._draw_fingertip_overlay(
            right_frame,
            expected=expected_right,
            detected=right_tip,
            label="Right",
            elapsed_s=elapsed_s,
        )
        if (
            left_tip is not None
            and right_tip is not None
            and elapsed_s >= FINGERTIP_STABILITY_SECONDS
        ):
            samples = list(self._fingertip_samples)
            left_radius = self._stability_radius([sample[1] for sample in samples])
            right_radius = self._stability_radius([sample[2] for sample in samples])
            stable_left = self._stable_point([sample[1] for sample in samples])
            stable_right = self._stable_point([sample[2] for sample in samples])
            if stable_left is None or stable_right is None:
                self._fingertip_samples.clear()
                self._fingertip_feedback = (
                    "Not stable: fingertip drift "
                    f"L {left_radius:.0f}px, R {right_radius:.0f}px "
                    f"(limit {FINGERTIP_MAX_STABILITY_RADIUS_PX:.0f}px). Restarting."
                )
            else:
                self._left_fingertips.append(stable_left)
                self._right_fingertips.append(stable_right)
                self._fingertip_corner_index += 1
                self._fingertip_samples.clear()
                self._fingertip_feedback = (
                    f"Captured corner {self._fingertip_corner_index}. "
                    + (
                        f"Move to corner {self._fingertip_corner_index + 1}."
                        if self._fingertip_corner_index < CORNER_COUNT
                        else "All corners captured."
                    )
                )
                if self._fingertip_corner_index == CORNER_COUNT:
                    if self._save_btn is not None:
                        self._save_btn.configure(state=tk.NORMAL)
                    if self._status_var is not None:
                        self._status_var.set(
                            "All four fingertips captured. Save the refined calibration."
                        )
                    return

        if left_tip is not None and right_tip is not None:
            if elapsed_s < FINGERTIP_STABILITY_SECONDS:
                self._fingertip_feedback = (
                    "Both fingertips detected. Keep still until the timer reaches 3.0 seconds."
                )
        elif (
            self._last_fingertip_detection_s is not None
            and now - self._last_fingertip_detection_s <= FINGERTIP_MAX_MISSING_SECONDS
        ):
            self._fingertip_feedback = "Brief detection loss tolerated; keep your finger still."
        else:
            self._fingertip_feedback = "Need an index fingertip detected near the highlighted corner in both feeds."

        self._show_stereo_frame(left_overlay, right_overlay)
        if self._status_var is not None:
            self._status_var.set(
                f"Refining corner {self._fingertip_corner_index + 1}/4 — "
                f"{self._fingertip_feedback}"
            )

        self._window.after(FINGERTIP_FEED_INTERVAL_MS, self._refresh_fingertip_feed)

    def _finish_fingertip_refinement(self) -> None:
        if self._rough_calibration is None:
            return
        try:
            calibration = refine_calibration_from_fingertips(
                self._rough_calibration,
                left_corners=[(float(x), float(y)) for x, y in self._left_corners],
                right_corners=[(float(x), float(y)) for x, y in self._right_corners],
                left_fingertips=self._left_fingertips,
                right_fingertips=self._right_fingertips,
            )
        except ValueError as exc:
            messagebox.showerror("Refinement failed", str(exc), parent=self._window)
            return
        calibration = save_calibration(calibration)
        if self._on_save is not None:
            self._on_save(calibration)
        self._close()

    def _close(self) -> None:
        if self._fingertip_detector is not None:
            self._fingertip_detector.close()
            self._fingertip_detector = None
        if self._window is not None:
            self._window.destroy()
