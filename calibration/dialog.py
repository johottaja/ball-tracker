from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from video_viewer.display import fit_size

from .homography import build_table_calibration
from .storage import load_calibration, save_calibration
from .types import TableCalibration

CORNER_COUNT = 4
# Bright blue, slightly dark (BGR).
MARKER_COLOR_BGR = (200, 120, 0)
MARKER_LINE_THICKNESS = 2


class TableCalibrationDialog:
    """Side-by-side table-corner clicking UI for stereo camera calibration."""

    def __init__(
        self,
        parent: tk.Tk,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        max_total_size: tuple[int, int],
        *,
        on_save: Callable[[TableCalibration], None] | None = None,
    ) -> None:
        self._parent = parent
        self._on_save = on_save
        self._left_base = left_frame.copy()
        self._right_base = right_frame.copy()
        self._max_total_size = max_total_size
        self._panel_size = fit_size(
            left_frame.shape[1],
            left_frame.shape[0],
            (max_total_size[0] // 2, max_total_size[1]),
        )
        self._left_corners: list[tuple[int, int]] = []
        self._right_corners: list[tuple[int, int]] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._window: tk.Toplevel | None = None
        self._save_btn: ttk.Button | None = None

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

        root = ttk.Frame(win, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text=(
                "Click four table corners on each feed in the same order (clockwise from "
                "above): 1 (+length,+width), 2 (+length,−width), 3 (−length,−width), "
                "4 (−length,+width). Origin is table center; length = +X, width = +Y. "
                "Use Reset to start over."
            ),
            wraplength=self._panel_size[0] * 2 - 24,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))

        self._status_var = tk.StringVar(value=self._status_text())
        ttk.Label(root, textvariable=self._status_var).pack(fill=tk.X, pady=(0, 8))

        self._image_label = ttk.Label(root, anchor=tk.CENTER)
        self._image_label.pack(fill=tk.BOTH, expand=True)
        self._image_label.bind("<Button-1>", self._on_click)

        dims = ttk.Frame(root)
        dims.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(dims, text="Table length (m):").pack(side=tk.LEFT)
        self._length_var = tk.StringVar(value=self._default_length)
        ttk.Entry(dims, textvariable=self._length_var, width=10).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(dims, text="Table width (m):").pack(side=tk.LEFT)
        self._width_var = tk.StringVar(value=self._default_width)
        ttk.Entry(dims, textvariable=self._width_var, width=10).pack(side=tk.LEFT, padx=(4, 0))

        focal_row = ttk.Frame(root)
        focal_row.pack(fill=tk.X, pady=(8, 0))
        self._match_right_focal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            focal_row,
            text="Right camera: use left focal length (for cropped/zoomed feeds)",
            variable=self._match_right_focal_var,
        ).pack(anchor=tk.W)

        hfov_row = ttk.Frame(root)
        hfov_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(hfov_row, text="Right horizontal FOV (°), optional:").pack(side=tk.LEFT)
        self._right_hfov_var = tk.StringVar(value="")
        ttk.Entry(hfov_row, textvariable=self._right_hfov_var, width=8).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            hfov_row,
            text="overrides the checkbox when set",
            foreground="#666666",
        ).pack(side=tk.LEFT, padx=(8, 0))

        btn_row = ttk.Frame(root)
        btn_row.pack(fill=tk.X, pady=(12, 0))

        self._save_btn = ttk.Button(btn_row, text="Save", command=self._save, state=tk.DISABLED)
        self._save_btn.pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Reset", command=self._reset_corners).pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 4))

        win.protocol("WM_DELETE_WINDOW", win.destroy)
        self._refresh_image()

    def _status_text(self) -> str:
        return (
            f"Left: {len(self._left_corners)}/{CORNER_COUNT} corners   "
            f"Right: {len(self._right_corners)}/{CORNER_COUNT} corners"
        )

    def _marker_radius(self, frame: np.ndarray) -> int:
        return max(6, min(frame.shape[0], frame.shape[1]) // 80)

    def _draw_overlay(self, frame: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
        out = frame.copy()
        if not corners:
            return out

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

    def _stereo_photo(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> ImageTk.PhotoImage:
        panel_size = self._panel_size
        left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        left_image = Image.fromarray(left_rgb).resize(panel_size, Image.Resampling.LANCZOS)
        right_image = Image.fromarray(right_rgb).resize(panel_size, Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (panel_size[0] * 2, panel_size[1]))
        combined.paste(left_image, (0, 0))
        combined.paste(right_image, (panel_size[0], 0))
        return ImageTk.PhotoImage(combined)

    def _refresh_image(self) -> None:
        left = self._draw_overlay(self._left_base, self._left_corners)
        right = self._draw_overlay(self._right_base, self._right_corners)
        self._photo = self._stereo_photo(left, right)
        self._image_label.configure(image=self._photo)

    def _display_to_image_coords(
        self,
        display_x: int,
        display_y: int,
        *,
        panel_index: int,
    ) -> tuple[int, int]:
        base = self._left_base if panel_index == 0 else self._right_base
        frame_h, frame_w = base.shape[:2]
        panel_w, panel_h = self._panel_size

        local_x = display_x if panel_index == 0 else display_x - panel_w
        x = int(local_x * frame_w / panel_w)
        y = int(display_y * frame_h / panel_h)
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        return x, y

    def _on_click(self, event: tk.Event) -> None:
        panel_w = self._panel_size[0]
        if event.x < panel_w:
            panel_index = 0
            corners = self._left_corners
        else:
            panel_index = 1
            corners = self._right_corners

        if len(corners) >= CORNER_COUNT:
            return

        point = self._display_to_image_coords(event.x, event.y, panel_index=panel_index)
        corners.append(point)
        self._status_var.set(self._status_text())
        self._refresh_image()
        self._update_save_state()

    def _update_save_state(self) -> None:
        complete = (
            len(self._left_corners) == CORNER_COUNT
            and len(self._right_corners) == CORNER_COUNT
        )
        state = tk.NORMAL if complete else tk.DISABLED
        if self._save_btn is not None:
            self._save_btn.configure(state=state)

    def _reset_corners(self) -> None:
        self._left_corners.clear()
        self._right_corners.clear()
        self._status_var.set(self._status_text())
        self._refresh_image()
        self._update_save_state()

    def _save(self) -> None:
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
        right_hfov_text = self._right_hfov_var.get().strip()
        right_horizontal_fov_deg: float | None = None
        if right_hfov_text:
            try:
                right_horizontal_fov_deg = float(right_hfov_text)
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    "Right horizontal FOV must be numeric when provided.",
                    parent=self._window,
                )
                return
            if right_horizontal_fov_deg <= 0.0 or right_horizontal_fov_deg >= 179.0:
                messagebox.showerror(
                    "Invalid input",
                    "Right horizontal FOV must be between 0° and 179°.",
                    parent=self._window,
                )
                return

        try:
            calibration = build_table_calibration(
                length_m=length_m,
                width_m=width_m,
                image_width=image_width,
                image_height=image_height,
                left_corners=[(float(x), float(y)) for x, y in self._left_corners],
                right_corners=[(float(x), float(y)) for x, y in self._right_corners],
                match_right_focal_to_left=self._match_right_focal_var.get(),
                right_horizontal_fov_deg=right_horizontal_fov_deg,
            )
        except ValueError as exc:
            messagebox.showerror("Calibration failed", str(exc), parent=self._window)
            return

        save_calibration(calibration)
        if self._on_save is not None:
            self._on_save(calibration)

        if self._window is not None:
            self._window.destroy()
