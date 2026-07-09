from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .layout import CameraLayoutInfo, layout_info_from_calibration
from .types import TableCalibration

CANVAS_SIZE = (520, 360)
CANVAS_PADDING_M = 0.35


class CameraLayoutDialog:
    """Top-down table + camera FOV visualization."""

    def __init__(
        self,
        parent: tk.Tk,
        calibration: TableCalibration | None,
    ) -> None:
        self._parent = parent
        self._calibration = calibration
        self._window: tk.Toplevel | None = None

    def show(self) -> None:
        if self._calibration is None:
            messagebox.showinfo(
                "No calibration",
                "Load or create a table calibration first.",
                parent=self._parent,
            )
            return

        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        calibration = self._calibration
        layouts = layout_info_from_calibration(calibration)
        if not layouts:
            messagebox.showerror(
                "Camera layout",
                "Could not derive camera positions from the calibration.",
                parent=self._parent,
            )
            return

        win = tk.Toplevel(self._parent)
        win.title("Camera layout")
        win.transient(self._parent)
        self._window = win

        root = ttk.Frame(win, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text=(
                "Top-down view: table centered at origin, +X along length, +Y along width. "
                "Red = left (main), blue = right."
            ),
            wraplength=CANVAS_SIZE[0],
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))

        canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE[0],
            height=CANVAS_SIZE[1],
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        canvas.pack()
        self._draw_scene(canvas, calibration, layouts)

        stats = ttk.Frame(root)
        stats.pack(fill=tk.X, pady=(12, 0))
        self._populate_stats(stats, calibration, layouts)

        ttk.Button(root, text="Close", command=win.destroy).pack(anchor=tk.E, pady=(12, 0))
        win.protocol("WM_DELETE_WINDOW", win.destroy)

    def _world_bounds(
        self,
        calibration: TableCalibration,
        layouts: list[CameraLayoutInfo],
    ) -> tuple[float, float, float, float]:
        half_length = calibration.table_length_m / 2.0
        half_width = calibration.table_width_m / 2.0
        min_x = -half_length
        max_x = half_length
        min_y = -half_width
        max_y = half_width

        for layout in layouts:
            min_x = min(min_x, layout.center[0])
            max_x = max(max_x, layout.center[0])
            min_y = min(min_y, layout.center[1])
            max_y = max(max_y, layout.center[1])
            if layout.fov_left_xy is not None:
                min_x = min(min_x, layout.fov_left_xy[0])
                max_x = max(max_x, layout.fov_left_xy[0])
                min_y = min(min_y, layout.fov_left_xy[1])
                max_y = max(max_y, layout.fov_left_xy[1])
            if layout.fov_right_xy is not None:
                min_x = min(min_x, layout.fov_right_xy[0])
                max_x = max(max_x, layout.fov_right_xy[0])
                min_y = min(min_y, layout.fov_right_xy[1])
                max_y = max(max_y, layout.fov_right_xy[1])

        pad = CANVAS_PADDING_M
        return min_x - pad, min_y - pad, max_x + pad, max_y + pad

    def _world_to_canvas(
        self,
        x: float,
        y: float,
        *,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        margin = 16.0
        scale = min(
            (CANVAS_SIZE[0] - 2 * margin) / span_x,
            (CANVAS_SIZE[1] - 2 * margin) / span_y,
        )
        center_x = CANVAS_SIZE[0] / 2.0
        center_y = CANVAS_SIZE[1] / 2.0
        world_center_x = (min_x + max_x) / 2.0
        world_center_y = (min_y + max_y) / 2.0
        canvas_x = center_x + (x - world_center_x) * scale
        canvas_y = center_y - (y - world_center_y) * scale
        return canvas_x, canvas_y

    def _draw_scene(
        self,
        canvas: tk.Canvas,
        calibration: TableCalibration,
        layouts: list[CameraLayoutInfo],
    ) -> None:
        bounds = self._world_bounds(calibration, layouts)
        half_length = calibration.table_length_m / 2.0
        half_width = calibration.table_width_m / 2.0

        corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]
        table_points: list[float] = []
        for x, y in corners:
            cx, cy = self._world_to_canvas(x, y, bounds=bounds)
            table_points.extend([cx, cy])
        canvas.create_polygon(*table_points, fill="black", outline="black")

        origin_x, origin_y = self._world_to_canvas(0.0, 0.0, bounds=bounds)
        canvas.create_oval(
            origin_x - 3,
            origin_y - 3,
            origin_x + 3,
            origin_y + 3,
            fill="#666666",
            outline="",
        )
        canvas.create_text(origin_x + 8, origin_y - 8, text="origin", fill="#666666", anchor=tk.W)

        for layout in layouts:
            cam_x, cam_y = layout.center[0], layout.center[1]
            cx, cy = self._world_to_canvas(cam_x, cam_y, bounds=bounds)
            if layout.fov_left_xy is not None:
                lx, ly = self._world_to_canvas(
                    layout.fov_left_xy[0],
                    layout.fov_left_xy[1],
                    bounds=bounds,
                )
                canvas.create_line(cx, cy, lx, ly, fill=layout.color, width=2)
            if layout.fov_right_xy is not None:
                rx, ry = self._world_to_canvas(
                    layout.fov_right_xy[0],
                    layout.fov_right_xy[1],
                    bounds=bounds,
                )
                canvas.create_line(cx, cy, rx, ry, fill=layout.color, width=2)

            canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=layout.color, outline="black")
            label = "left" if layout.name == "left" else layout.name
            canvas.create_text(cx + 10, cy - 10, text=label, fill=layout.color, anchor=tk.W)

    def _populate_stats(
        self,
        parent: ttk.Frame,
        calibration: TableCalibration,
        layouts: list[CameraLayoutInfo],
    ) -> None:
        ttk.Label(
            parent,
            text=(
                f"Table: {calibration.table_length_m:.2f} m × "
                f"{calibration.table_width_m:.2f} m   "
                f"Image: {calibration.image_width}×{calibration.image_height}"
            ),
        ).pack(anchor=tk.W)

        for layout in layouts:
            side = "Left (main)" if layout.name == "left" else "Right"
            ttk.Label(
                parent,
                text=(
                    f"{side}: XY distance {layout.xy_distance_m:.2f} m, "
                    f"Z {layout.z_m:+.2f} m, "
                    f"yaw {layout.yaw_deg:+.1f}°, "
                    f"pitch {layout.pitch_deg:+.1f}°, "
                    f"horizontal FOV {layout.horizontal_fov_deg:.1f}°"
                ),
            ).pack(anchor=tk.W, pady=(4, 0))

        left = next((item for item in layouts if item.name == "left"), None)
        right = next((item for item in layouts if item.name == "right"), None)
        stereo = calibration.layout.stereo if calibration.layout is not None else None
        if stereo is not None:
            ttk.Label(
                parent,
                text=(
                    f"Baseline: {stereo.baseline_xy_m:.2f} m (XY), "
                    f"{stereo.baseline_3d_m:.2f} m (3D), "
                    f"ΔZ {stereo.delta_z_m:+.2f} m"
                ),
            ).pack(anchor=tk.W, pady=(8, 0))
        elif left is not None and right is not None:
            dx = right.center[0] - left.center[0]
            dy = right.center[1] - left.center[1]
            dz = right.center[2] - left.center[2]
            baseline_xy = (dx * dx + dy * dy) ** 0.5
            baseline_3d = (dx * dx + dy * dy + dz * dz) ** 0.5
            ttk.Label(
                parent,
                text=(
                    f"Baseline: {baseline_xy:.2f} m (XY), {baseline_3d:.2f} m (3D), "
                    f"ΔZ {dz:+.2f} m"
                ),
            ).pack(anchor=tk.W, pady=(8, 0))
