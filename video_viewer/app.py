from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from .camera import CameraDevice, CameraReader, configure_camera_fps, open_camera, probe_cameras
from .config import DEFAULT_VIDEO, RECORDINGS_DIR, TARGET_RECORD_FPS
from .display import fit_size
from .filter_controls import FilterControls
from .filters import FrameFilter
from .playback import (
    filter_inputs_for_playback,
    frame_to_display_photo,
    read_frame_at,
    uses_gru_streaming,
)
from .recording import create_writer


class VideoViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Recorder & Frame Viewer")
        self.root.minsize(640, 520)

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self.mode = tk.StringVar(value="record")
        self.video_path = tk.StringVar(value=str(DEFAULT_VIDEO))

        self.cap: cv2.VideoCapture | None = None
        self.camera_reader: CameraReader | None = None
        self.writer: cv2.VideoWriter | None = None
        self._preview_frame_id = 0
        self.recording = False
        self.playing = False
        self.frame_index = 0
        self.frame_count = 0
        self.fps = TARGET_RECORD_FPS
        self.record_fps = TARGET_RECORD_FPS
        self.frame_photo = None
        self.after_id: str | None = None
        self.display_size = fit_size(640, 480)
        self.camera_index = 0
        self.cameras: list[CameraDevice] = []
        self.frame_filter = FrameFilter()
        self._last_raw_frame: np.ndarray | None = None
        self._gru_stream_frame_index: int | None = None

        self._build_ui()
        self._build_menu()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._enter_record_mode()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar,
            text="Record",
            variable=self.mode,
            value="record",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(
            toolbar,
            text="Playback",
            variable=self.mode,
            value="playback",
            command=self._on_mode_change,
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(toolbar, text="Open video…", command=self._open_video).pack(
            side=tk.RIGHT
        )

        self.filter_controls = FilterControls(
            self.root,
            on_change=self._on_filter_change,
        )

        self.video_label = ttk.Label(self.root, text="No video", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Ready to record.")
        ttk.Label(self.root, textvariable=self.status_var).pack(
            fill=tk.X, padx=8, pady=(0, 4)
        )

        self.record_controls = ttk.Frame(self.root, padding=8)

        camera_row = ttk.Frame(self.record_controls)
        camera_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(camera_row, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar()
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
            self.record_controls,
            text="Start recording",
            command=self._toggle_recording,
        )
        self.record_btn.pack()

        self.playback_controls = ttk.Frame(self.root, padding=8)
        btn_row = ttk.Frame(self.playback_controls)
        btn_row.pack()
        ttk.Button(btn_row, text="|◀ Beginning", command=self._go_to_start).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="◀ Frame", command=self._step_backward).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="Play", command=self._play).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Pause", command=self._pause).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_row, text="Frame ▶", command=self._step_forward).pack(
            side=tk.LEFT, padx=2
        )

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.filter_controls.add_menu(menubar)

    def _on_filter_change(self) -> None:
        self._gru_stream_frame_index = None
        self.frame_filter.set_filter(self.filter_controls.selected_filter_id())
        self.filter_controls.sync_combo_from_var()
        self._refresh_visible_frame()

    def _refresh_visible_frame(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return
        if self.mode.get() == "playback":
            self._show_frame_at(self.frame_index)
        elif self._last_raw_frame is not None:
            self._display_frame(self._last_raw_frame, previous_frame=None)

    def _on_mode_change(self) -> None:
        if self.mode.get() == "record":
            self._enter_record_mode()
        else:
            self._enter_playback_mode()

    def _cancel_after(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _release_capture(self) -> None:
        self._cancel_after()
        self._gru_stream_frame_index = None
        self.frame_filter.reset()
        self.playing = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.camera_reader is not None:
            self.camera_reader.stop()
            self.camera_reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._preview_frame_id = 0

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.camera_combo.configure(state=state)
        self.refresh_cameras_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _camera_label(self, index: int) -> str:
        for camera in self.cameras:
            if camera.index == index:
                return camera.label
        return f"Camera {index}"

    def _camera_name(self, index: int) -> str:
        for camera in self.cameras:
            if camera.index == index:
                return camera.name
        return f"Camera {index}"

    def _refresh_cameras(self) -> None:
        if self.recording:
            return

        self.cameras = probe_cameras()
        labels = [camera.label for camera in self.cameras]

        if not labels:
            self._cancel_after()
            if self.camera_reader is not None:
                self.camera_reader.stop()
                self.camera_reader = None
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_combo.configure(values=[])
            self.camera_var.set("")
            self.video_label.configure(image="", text="No camera found")
            self.status_var.set("No cameras detected. Connect one and press Refresh.")
            return

        self.camera_combo.configure(values=labels)
        camera_indices = [camera.index for camera in self.cameras]
        if self.camera_index not in camera_indices:
            self.camera_index = camera_indices[0]
        self.camera_var.set(self._camera_label(self.camera_index))
        self._open_camera(self.camera_index)

    def _on_camera_selected(self, _event: object | None = None) -> None:
        if self.recording:
            return
        label = self.camera_var.get()
        index = next(
            (camera.index for camera in self.cameras if camera.label == label),
            None,
        )
        if index is None:
            return
        if index == self.camera_index and self.cap is not None and self.cap.isOpened():
            return
        self.camera_index = index
        self._open_camera(index)

    def _open_camera(self, index: int) -> bool:
        self._cancel_after()
        if self.camera_reader is not None:
            self.camera_reader.stop()
            self.camera_reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._preview_frame_id = 0

        self.cap = open_camera(index)
        if not self.cap.isOpened():
            camera_name = self._camera_name(index)
            messagebox.showerror(
                "Camera error",
                f"Could not open {camera_name}. Check permissions and try again.",
            )
            self.status_var.set(f"{camera_name} unavailable.")
            return False

        self.camera_index = index
        self.record_fps = configure_camera_fps(self.cap)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_size = fit_size(width, height)
        self.camera_reader = CameraReader(self.cap)
        self.camera_reader.start()
        self.status_var.set(
            f"{self._camera_name(index)} — live preview @ {self.record_fps:.0f} fps. "
            f"Press Start recording to save to {DEFAULT_VIDEO.name}"
        )
        self._schedule_record_preview()
        return True

    def _on_captured_frame(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def _enter_record_mode(self) -> None:
        self._release_capture()
        self.recording = False
        self.record_btn.configure(text="Start recording")
        self._set_camera_controls_enabled(True)
        self.playback_controls.pack_forget()
        self.record_controls.pack(fill=tk.X)
        self._refresh_cameras()

    def _enter_playback_mode(self) -> None:
        self._release_capture()
        self.recording = False
        self.record_controls.pack_forget()
        self.playback_controls.pack(fill=tk.X)

        path = Path(self.video_path.get())
        if not path.is_file():
            self.status_var.set("No video loaded. Record one or use Open video…")
            self.video_label.configure(image="", text="No video loaded")
            return

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("Playback error", f"Could not open:\n{path}")
            self.status_var.set("Failed to open video.")
            return

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_size = fit_size(width, height)
        self.frame_index = 0
        self._show_frame_at(0)
        self._update_status()

    def _schedule_record_preview(self) -> None:
        if self.mode.get() != "record" or self.camera_reader is None:
            return
        ok, frame, frame_id = self.camera_reader.get_latest_frame()
        if ok and frame is not None and frame_id != self._preview_frame_id:
            self._preview_frame_id = frame_id
            self._last_raw_frame = frame
            self._display_frame(frame, previous_frame=None)
        delay = max(1, int(1000 / self.record_fps))
        self.after_id = self.root.after(delay, self._schedule_record_preview)

    def _toggle_recording(self) -> None:
        if self.camera_reader is None:
            return

        if not self.recording:
            width = self.camera_reader.frame_width
            height = self.camera_reader.frame_height
            self.writer = create_writer(self.record_fps, width, height)
            if not self.writer.isOpened():
                messagebox.showerror(
                    "Recording error", f"Could not create {DEFAULT_VIDEO}"
                )
                self.writer = None
                return
            self.recording = True
            if self.camera_reader is not None:
                self.camera_reader.set_frame_consumer(self._on_captured_frame)
            self.record_btn.configure(text="Stop recording")
            self._set_camera_controls_enabled(False)
            self.status_var.set(
                f"Recording to {DEFAULT_VIDEO.name} @ {self.record_fps:.0f} fps…"
            )
        else:
            self.recording = False
            if self.camera_reader is not None:
                self.camera_reader.set_frame_consumer(None)
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.record_btn.configure(text="Start recording")
            self._set_camera_controls_enabled(True)
            self.video_path.set(str(DEFAULT_VIDEO))
            self.status_var.set(
                f"Saved {DEFAULT_VIDEO.name}. Switch to Playback to review."
            )

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video",
            initialdir=RECORDINGS_DIR,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.video_path.set(path)
        self.mode.set("playback")
        self._enter_playback_mode()

    def _show_frame_at(self, index: int) -> bool:
        if self.cap is None:
            return False
        index = max(0, index)
        if self.frame_count > 0:
            index = min(index, self.frame_count - 1)

        ok, frame = read_frame_at(self.cap, index)
        if not ok or frame is None:
            return False

        self.frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if self.frame_index < 0:
            self.frame_index = index

        previous, window_frames, warmup_frames = filter_inputs_for_playback(
            self.cap,
            self.frame_filter,
            self.frame_index,
            self._gru_stream_frame_index,
        )

        self._display_frame(
            frame,
            previous_frame=previous,
            window_frames=window_frames,
            warmup_frames=warmup_frames,
        )
        if uses_gru_streaming(self.frame_filter):
            self._gru_stream_frame_index = self.frame_index
        self._update_status()
        return True

    def _display_frame(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        window_frames: list[np.ndarray] | None = None,
        warmup_frames: list[np.ndarray] | None = None,
    ) -> None:
        video_fps = self.fps if self.mode.get() == "playback" else None
        self.frame_photo = frame_to_display_photo(
            self.frame_filter,
            frame,
            self.display_size,
            previous_frame=previous_frame,
            window_frames=window_frames,
            warmup_frames=warmup_frames,
            video_fps=video_fps,
        )
        self.video_label.configure(image=self.frame_photo, text="")

    def _update_status(self) -> None:
        total = self.frame_count if self.frame_count > 0 else "?"
        name = Path(self.video_path.get()).name
        time_s = self.frame_index / self.fps if self.fps else 0
        self.status_var.set(
            f"{name} — frame {self.frame_index + 1} / {total} "
            f"({time_s:.2f}s @ {self.fps:.1f} fps)"
        )

    def _go_to_start(self) -> None:
        self._pause()
        self._show_frame_at(0)

    def _step_backward(self) -> None:
        self._pause()
        self._show_frame_at(self.frame_index - 1)

    def _step_forward(self) -> None:
        self._pause()
        self._show_frame_at(self.frame_index + 1)

    def _play(self) -> None:
        if self.cap is None or self.mode.get() != "playback":
            return
        if self.frame_count and self.frame_index >= self.frame_count - 1:
            self._show_frame_at(0)
        self.playing = True
        self._schedule_playback()

    def _pause(self) -> None:
        self.playing = False
        self._cancel_after()

    def _schedule_playback(self) -> None:
        if not self.playing or self.cap is None:
            return

        next_index = self.frame_index + 1
        if self.frame_count and next_index >= self.frame_count:
            self.playing = False
            self._show_frame_at(self.frame_count - 1)
            self.status_var.set(self.status_var.get() + " — end of video")
            return

        if not self._show_frame_at(next_index):
            self.playing = False
            return

        self.after_id = self.root.after(1, self._schedule_playback)

    def _on_close(self) -> None:
        self._release_capture()
        self.root.destroy()
