from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from .setup_config import CameraSetup, save_setup_config


class CameraSetupDialog:
    """Modal popup for editing camera geometry."""

    def __init__(
        self,
        parent: tk.Tk,
        setup: CameraSetup,
        *,
        on_save: Callable[[CameraSetup], None],
    ) -> None:
        self._parent = parent
        self._setup = setup
        self._on_save = on_save
        self._window: tk.Toplevel | None = None
        self._vars: dict[str, tk.StringVar] = {}

    def show(self) -> None:
        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        win = tk.Toplevel(self._parent)
        win.title("Camera setup")
        win.transient(self._parent)
        win.grab_set()
        self._window = win

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        fields = [
            ("main_distance_m", "Main camera distance from table center (m):"),
            ("secondary_distance_m", "Secondary camera distance from center (m):"),
            ("main_height_m", "Main camera height above table (m):"),
            ("secondary_height_m", "Secondary camera height above table (m):"),
            ("camera_angle_deg", "Angle between cameras (degrees):"),
            ("horizontal_fov_deg", "Assumed horizontal field of view (degrees):"),
        ]

        for key, label in fields:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=label, width=42).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(getattr(self._setup, key)))
            self._vars[key] = var
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.RIGHT)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_row, text="Save", command=self._save).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", win.destroy)

    def _save(self) -> None:
        try:
            setup = CameraSetup(
                main_distance_m=float(self._vars["main_distance_m"].get()),
                secondary_distance_m=float(self._vars["secondary_distance_m"].get()),
                main_height_m=float(self._vars["main_height_m"].get()),
                secondary_height_m=float(self._vars["secondary_height_m"].get()),
                camera_angle_deg=float(self._vars["camera_angle_deg"].get()),
                horizontal_fov_deg=float(self._vars["horizontal_fov_deg"].get()),
            )
        except ValueError:
            messagebox.showerror("Invalid input", "All fields must be numeric.", parent=self._window)
            return

        if setup.main_distance_m <= 0 or setup.secondary_distance_m <= 0:
            messagebox.showerror(
                "Invalid input",
                "Camera distances must be positive.",
                parent=self._window,
            )
            return
        if setup.main_height_m <= 0 or setup.secondary_height_m <= 0:
            messagebox.showerror(
                "Invalid input",
                "Camera heights must be positive.",
                parent=self._window,
            )
            return
        if setup.horizontal_fov_deg <= 0 or setup.horizontal_fov_deg >= 180:
            messagebox.showerror(
                "Invalid input",
                "Horizontal FOV must be between 0 and 180 degrees.",
                parent=self._window,
            )
            return

        save_setup_config(setup)
        self._on_save(setup)
        if self._window is not None:
            self._window.destroy()
