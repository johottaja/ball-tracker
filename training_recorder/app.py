from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import ImageTk

from video_viewer.camera import configure_camera_fps, open_camera, probe_cameras
from video_viewer.display import fit_size, frame_to_photo
from video_viewer.recording import create_writer

from .config import RECORDINGS_DIR, TARGET_RECORD_FPS
from .paths import next_clip_path, sanitize_training_set_name, training_set_dir


class TrainingRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Training Clip Recorder")
        self.root.minsize(640, 520)

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.recording = False
        self.record_fps = TARGET_RECORD_FPS
        self.frame_photo: ImageTk.PhotoImage | None = None
        self.after_id: str | None = None
        self.display_size = fit_size(640, 480)
        self.camera_index = 0
        self.camera_indices: list[int] = []
        self._last_raw_frame: np.ndarray | None = None
        self._current_clip_path: Path | None = None
        self.clips_recorded = 0

        self.training_set_var = tk.StringVar(value="")
        self.camera_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_cameras()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)

        set_row = ttk.Frame(controls)
        set_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(set_row, text="Training set:").pack(side=tk.LEFT)
        self.training_set_entry = ttk.Entry(
            set_row, textvariable=self.training_set_var, width=32
        )
        self.training_set_entry.pack(side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True)

        camera_row = ttk.Frame(controls)
        camera_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(camera_row, text="Camera:").pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_var,
            state="readonly",
            width=24,
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(4, 4))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)
        self.refresh_cameras_btn = ttk.Button(
            camera_row, text="Refresh", command=self._refresh_cameras
        )
        self.refresh_cameras_btn.pack(side=tk.LEFT)

        self.record_btn = ttk.Button(
            controls,
            text="Start clip",
            command=self._toggle_recording,
        )
        self.record_btn.pack()

        self.video_label = ttk.Label(self.root, text="No camera", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(self.root, textvariable=self.status_var).pack(
            fill=tk.X, padx=8, pady=(0, 8)
        )

    def _cancel_after(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _release_capture(self) -> None:
        self._cancel_after()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.camera_combo.configure(state=state)
        self.refresh_cameras_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.training_set_entry.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    @staticmethod
    def _camera_label(index: int) -> str:
        return f"Camera {index}"

    def _active_training_set(self) -> str:
        return sanitize_training_set_name(self.training_set_var.get())

    def _refresh_cameras(self) -> None:
        if self.recording:
            return

        self.camera_indices = probe_cameras()
        labels = [self._camera_label(i) for i in self.camera_indices]

        if not labels:
            self._cancel_after()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_combo.configure(values=[])
            self.camera_var.set("")
            self.video_label.configure(image="", text="No camera found")
            self.status_var.set("No cameras detected. Connect one and press Refresh.")
            return

        self.camera_combo.configure(values=labels)
        if self.camera_index not in self.camera_indices:
            self.camera_index = self.camera_indices[0]
        self.camera_var.set(self._camera_label(self.camera_index))
        self._open_camera(self.camera_index)

    def _on_camera_selected(self, _event: object | None = None) -> None:
        if self.recording:
            return
        label = self.camera_var.get()
        try:
            index = int(label.removeprefix("Camera "))
        except ValueError:
            return
        if index == self.camera_index and self.cap is not None and self.cap.isOpened():
            return
        self.camera_index = index
        self._open_camera(index)

    def _open_camera(self, index: int) -> bool:
        self._cancel_after()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = open_camera(index)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera error",
                f"Could not open camera {index}. Check permissions and try again.",
            )
            self.status_var.set(f"Camera {index} unavailable.")
            return False

        self.camera_index = index
        self.record_fps = configure_camera_fps(self.cap)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_size = fit_size(width, height)
        set_name = self._active_training_set()
        self.status_var.set(
            f"Camera {index} — live preview @ {self.record_fps:.0f} fps. "
            f"Clips save to recordings/{set_name}/"
        )
        self._schedule_preview()
        return True

    def _schedule_preview(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self._last_raw_frame = frame
            if self.recording and self.writer is not None:
                self.writer.write(frame)
            self._display_frame(frame)
        delay = max(1, int(1000 / self.record_fps))
        self.after_id = self.root.after(delay, self._schedule_preview)

    def _display_frame(self, frame: np.ndarray) -> None:
        self.frame_photo = frame_to_photo(frame, self.display_size)
        self.video_label.configure(image=self.frame_photo, text="")

    def _toggle_recording(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return

        if not self.recording:
            clip_path = next_clip_path(self.training_set_var.get())
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = create_writer(self.record_fps, width, height, path=clip_path)
            if not self.writer.isOpened():
                messagebox.showerror("Recording error", f"Could not create {clip_path}")
                self.writer = None
                return
            self.recording = True
            self._current_clip_path = clip_path
            self.record_btn.configure(text="Stop clip")
            self._set_controls_enabled(False)
            set_name = self._active_training_set()
            self.status_var.set(
                f"Recording clip to recordings/{set_name}/{clip_path.name} "
                f"@ {self.record_fps:.0f} fps…"
            )
        else:
            self.recording = False
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.record_btn.configure(text="Start clip")
            self._set_controls_enabled(True)
            self.clips_recorded += 1
            saved = self._current_clip_path
            self._current_clip_path = None
            set_dir = training_set_dir(self.training_set_var.get())
            if saved is not None:
                self.status_var.set(
                    f"Saved {saved.name} ({self.clips_recorded} clip(s) this session). "
                    f"Folder: {set_dir.relative_to(RECORDINGS_DIR.parent)}/"
                )
            self._open_camera(self.camera_index)

    def _on_close(self) -> None:
        self._release_capture()
        self.root.destroy()
