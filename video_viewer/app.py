from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import ImageTk

from .camera import configure_camera_fps, open_camera, probe_cameras
from .config import DEFAULT_VIDEO, FRAME_WINDOW_SIZE, RECORDINGS_DIR, TARGET_RECORD_FPS
from .display import fit_size, frame_to_photo
from .filters import FILTER_LABELS, FilterId, FrameFilter
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
        self.writer: cv2.VideoWriter | None = None
        self.recording = False
        self.playing = False
        self.frame_index = 0
        self.frame_count = 0
        self.fps = TARGET_RECORD_FPS
        self.record_fps = TARGET_RECORD_FPS
        self.frame_photo: ImageTk.PhotoImage | None = None
        self.after_id: str | None = None
        self.display_size = fit_size(640, 480)
        self.camera_index = 0
        self.camera_indices: list[int] = []
        self.frame_filter = FrameFilter()
        self.filter_var = tk.StringVar(value=FilterId.NONE.value)
        self._last_raw_frame: np.ndarray | None = None

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

        filter_row = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        filter_row.pack(fill=tk.X)
        ttk.Label(filter_row, text="Filter:").pack(side=tk.LEFT)
        self.filter_combo = ttk.Combobox(
            filter_row,
            state="readonly",
            width=42,
        )
        self.filter_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_combo)
        self._sync_filter_combo_labels()
        self.filter_combo.set(FILTER_LABELS[FilterId.NONE])

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

        filters_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filters", menu=filters_menu)
        for filter_id in FilterId:
            filters_menu.add_radiobutton(
                label=FILTER_LABELS[filter_id],
                variable=self.filter_var,
                value=filter_id.value,
                command=self._on_filter_change,
            )

    def _sync_filter_combo_labels(self) -> None:
        self._filter_value_by_label = {
            FILTER_LABELS[fid]: fid.value for fid in FilterId
        }
        self._filter_label_by_value = {
            value: label for label, value in self._filter_value_by_label.items()
        }
        self.filter_combo.configure(values=list(self._filter_value_by_label))

    def _on_filter_combo(self, _event: object | None = None) -> None:
        label = self.filter_combo.get()
        value = self._filter_value_by_label.get(label)
        if value is not None:
            self.filter_var.set(value)
            self._on_filter_change()

    def _on_filter_change(self) -> None:
        self.frame_filter.set_filter(FilterId(self.filter_var.get()))
        label = self._filter_label_by_value.get(self.filter_var.get(), "")
        if label and self.filter_combo.get() != label:
            self.filter_combo.set(label)
        self._refresh_visible_frame()

    def _refresh_visible_frame(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return
        if self.mode.get() == "playback":
            self._show_frame_at(self.frame_index)
        elif self._last_raw_frame is not None:
            self._display_frame(self._last_raw_frame, previous_frame=None)

    def _previous_frame_for_diff(self, index: int) -> np.ndarray | None:
        if index <= 0:
            return None
        ok, frame = self._read_frame_at(index - 1)
        return frame if ok else None

    def _window_frames_for_diff(self, index: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        start = max(0, index - FRAME_WINDOW_SIZE)
        for i in range(start, index):
            ok, frame = self._read_frame_at(i)
            if ok:
                frames.append(frame)
        return frames

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
        self.frame_filter.reset()
        self.playing = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.camera_combo.configure(state=state)
        self.refresh_cameras_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    @staticmethod
    def _camera_label(index: int) -> str:
        return f"Camera {index}"

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
        self.status_var.set(
            f"Camera {index} — live preview @ {self.record_fps:.0f} fps. "
            f"Press Start recording to save to {DEFAULT_VIDEO.name}"
        )
        self._schedule_record_preview()
        return True

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
        if self.mode.get() != "record" or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self._last_raw_frame = frame
            if self.recording and self.writer is not None:
                self.writer.write(frame)
            self._display_frame(frame, previous_frame=None)
        delay = max(1, int(1000 / self.record_fps))
        self.after_id = self.root.after(delay, self._schedule_record_preview)

    def _toggle_recording(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return

        if not self.recording:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = create_writer(self.record_fps, width, height)
            if not self.writer.isOpened():
                messagebox.showerror(
                    "Recording error", f"Could not create {DEFAULT_VIDEO}"
                )
                self.writer = None
                return
            self.recording = True
            self.record_btn.configure(text="Stop recording")
            self._set_camera_controls_enabled(False)
            self.status_var.set(
                f"Recording to {DEFAULT_VIDEO.name} @ {self.record_fps:.0f} fps…"
            )
        else:
            self.recording = False
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

    def _read_frame_at(self, index: int) -> tuple[bool, object]:
        if self.cap is None:
            return False, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.cap.read()

    def _show_frame_at(self, index: int) -> bool:
        if self.cap is None:
            return False
        index = max(0, index)
        if self.frame_count > 0:
            index = min(index, self.frame_count - 1)

        ok, frame = self._read_frame_at(index)
        if not ok:
            return False

        self.frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if self.frame_index < 0:
            self.frame_index = index

        previous = None
        window_frames = None
        if self.frame_filter.filter_id in (
            FilterId.FRAME_DIFF,
            FilterId.DETECTION,
        ):
            previous = self._previous_frame_for_diff(self.frame_index)
        elif self.frame_filter.filter_id == FilterId.FRAME_DIFF_WINDOW:
            window_frames = self._window_frames_for_diff(self.frame_index)

        self._display_frame(
            frame,
            previous_frame=previous,
            window_frames=window_frames,
        )
        self._update_status()
        return True

    def _display_frame(
        self,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        window_frames: list[np.ndarray] | None = None,
    ) -> None:
        filtered = self.frame_filter.apply(
            frame,
            previous_frame=previous_frame,
            window_frames=window_frames,
        )
        self.frame_photo = frame_to_photo(filtered, self.display_size)
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

        delay = max(1, int(1000 / self.fps))
        self.after_id = self.root.after(delay, self._schedule_playback)

    def _on_close(self) -> None:
        self._release_capture()
        self.root.destroy()
