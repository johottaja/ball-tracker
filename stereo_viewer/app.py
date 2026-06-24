from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from video_viewer.camera import (
    CameraDevice,
    CameraReader,
    configure_camera_fps,
    open_camera,
    probe_cameras,
)
from video_viewer.filter_controls import FilterControls
from video_viewer.filters import FrameFilter
from video_viewer.filters import FilterId
from video_viewer.playback import (
    filter_inputs_for_playback,
    gru_warmup_frames_if_needed,
    previous_frame_for_diff,
    read_frame_at,
    uses_gru_streaming,
)
from video_viewer.recording import create_writer

from .config import LEFT_VIDEO, RECORDINGS_DIR, RIGHT_VIDEO, STEREO_DISPLAY_MAX_SIZE, TARGET_FPS
from .display import panel_size_for_frame, stereo_frame_to_photo
from .stereo_tracking import StereoTrackingProcessor


@dataclass
class CameraStream:
    label: str
    default_video: Path
    camera_index: int = 0
    cap: cv2.VideoCapture | None = None
    camera_reader: CameraReader | None = None
    writer: cv2.VideoWriter | None = None
    frame_filter: FrameFilter = field(default_factory=FrameFilter)
    gru_stream_frame_index: int | None = None
    last_raw_frame: np.ndarray | None = None
    preview_frame_id: int = 0
    video_path: Path | None = None

    def __post_init__(self) -> None:
        if self.video_path is None:
            self.video_path = self.default_video


class StereoViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stereo Video Recorder & Frame Viewer")
        self.root.minsize(960, 520)

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self.mode = tk.StringVar(value="record")
        self.left = CameraStream("Left", LEFT_VIDEO)
        self.right = CameraStream("Right", RIGHT_VIDEO)

        self.recording = False
        self.playing = False
        self.frame_index = 0
        self.frame_count = 0
        self.fps = TARGET_FPS
        self.record_fps = TARGET_FPS
        self.frame_photo = None
        self.after_id: str | None = None
        self.panel_size = panel_size_for_frame(640, 480, STEREO_DISPLAY_MAX_SIZE)
        self.cameras: list[CameraDevice] = []
        self.stereo_tracking = StereoTrackingProcessor()

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

        ttk.Button(toolbar, text="Open stereo pair…", command=self._open_videos).pack(
            side=tk.RIGHT
        )

        self.filter_controls = FilterControls(
            self.root,
            on_change=self._on_filter_change,
            include_stereo=True,
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

        ttk.Label(camera_row, text="Camera 1 (left):").pack(side=tk.LEFT)
        self.left_camera_var = tk.StringVar()
        self.left_camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.left_camera_var,
            state="readonly",
            width=22,
        )
        self.left_camera_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.left_camera_combo.bind("<<ComboboxSelected>>", self._on_left_camera_selected)

        ttk.Label(camera_row, text="Camera 2 (right):").pack(side=tk.LEFT)
        self.right_camera_var = tk.StringVar()
        self.right_camera_combo = ttk.Combobox(
            camera_row,
            textvariable=self.right_camera_var,
            state="readonly",
            width=22,
        )
        self.right_camera_combo.pack(side=tk.LEFT, padx=(4, 4))
        self.right_camera_combo.bind(
            "<<ComboboxSelected>>", self._on_right_camera_selected
        )
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

    def _streams(self) -> tuple[CameraStream, CameraStream]:
        return self.left, self.right

    def _is_stereo_tracking(self) -> bool:
        return self.filter_controls.selected_filter_id() == FilterId.STEREO_TRACKING

    def _on_filter_change(self) -> None:
        filter_id = self.filter_controls.selected_filter_id()
        self.stereo_tracking.reset()
        for stream in self._streams():
            stream.gru_stream_frame_index = None
            if filter_id == FilterId.STEREO_TRACKING:
                stream.frame_filter.set_filter(FilterId.NONE)
            else:
                stream.frame_filter.set_filter(filter_id)
        self.filter_controls.sync_combo_from_var()
        self._refresh_visible_frame()

    def _refresh_visible_frame(self) -> None:
        if self.mode.get() == "playback":
            if self._both_caps_open():
                self._show_frame_at(self.frame_index)
        elif self.left.last_raw_frame is not None and self.right.last_raw_frame is not None:
            self._display_stereo_frames(
                self.left.last_raw_frame,
                self.right.last_raw_frame,
            )

    def _both_caps_open(self) -> bool:
        return (
            self.left.cap is not None
            and self.left.cap.isOpened()
            and self.right.cap is not None
            and self.right.cap.isOpened()
        )

    def _cameras_ready_for_record(self) -> bool:
        return (
            self.left.camera_reader is not None
            and self.right.camera_reader is not None
        )

    def _on_mode_change(self) -> None:
        if self.mode.get() == "record":
            self._enter_record_mode()
        else:
            self._enter_playback_mode()

    def _cancel_after(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _release_stream(self, stream: CameraStream) -> None:
        stream.gru_stream_frame_index = None
        stream.frame_filter.reset()
        if stream.writer is not None:
            stream.writer.release()
            stream.writer = None
        if stream.camera_reader is not None:
            stream.camera_reader.stop()
            stream.camera_reader = None
        if stream.cap is not None:
            stream.cap.release()
            stream.cap = None
        stream.preview_frame_id = 0
        stream.last_raw_frame = None

    def _release_capture(self) -> None:
        self._cancel_after()
        self.playing = False
        self.stereo_tracking.reset()
        for stream in self._streams():
            self._release_stream(stream)

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.left_camera_combo.configure(state=state)
        self.right_camera_combo.configure(state=state)
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

    def _index_for_label(self, label: str) -> int | None:
        return next(
            (camera.index for camera in self.cameras if camera.label == label),
            None,
        )

    def _refresh_cameras(self) -> None:
        if self.recording:
            return

        self.cameras = probe_cameras()
        labels = [camera.label for camera in self.cameras]

        if len(labels) < 2:
            self._cancel_after()
            for stream in self._streams():
                self._release_stream(stream)
            self.left_camera_combo.configure(values=labels)
            self.right_camera_combo.configure(values=labels)
            self.left_camera_var.set(labels[0] if labels else "")
            self.right_camera_var.set("")
            self.video_label.configure(image="", text="Need two cameras")
            if not labels:
                self.status_var.set(
                    "No cameras detected. Connect two cameras and press Refresh."
                )
            else:
                self.status_var.set(
                    "Only one camera found. Connect a second camera and press Refresh."
                )
            return

        self.left_camera_combo.configure(values=labels)
        self.right_camera_combo.configure(values=labels)
        indices = [camera.index for camera in self.cameras]
        if self.left.camera_index not in indices:
            self.left.camera_index = indices[0]
        if (
            self.right.camera_index not in indices
            or self.right.camera_index == self.left.camera_index
        ):
            self.right.camera_index = indices[1] if indices[1] != indices[0] else indices[0]

        self.left_camera_var.set(self._camera_label(self.left.camera_index))
        self.right_camera_var.set(self._camera_label(self.right.camera_index))
        self._open_cameras()

    def _on_left_camera_selected(self, _event: object | None = None) -> None:
        if self.recording:
            return
        index = self._index_for_label(self.left_camera_var.get())
        if index is None:
            return
        if index == self.right.camera_index:
            messagebox.showwarning(
                "Camera conflict",
                "Camera 1 and Camera 2 must be different devices.",
            )
            self.left_camera_var.set(self._camera_label(self.left.camera_index))
            return
        if index == self.left.camera_index and self.left.cap is not None:
            return
        self.left.camera_index = index
        self._open_cameras()

    def _on_right_camera_selected(self, _event: object | None = None) -> None:
        if self.recording:
            return
        index = self._index_for_label(self.right_camera_var.get())
        if index is None:
            return
        if index == self.left.camera_index:
            messagebox.showwarning(
                "Camera conflict",
                "Camera 1 and Camera 2 must be different devices.",
            )
            self.right_camera_var.set(self._camera_label(self.right.camera_index))
            return
        if index == self.right.camera_index and self.right.cap is not None:
            return
        self.right.camera_index = index
        self._open_cameras()

    def _open_cameras(self) -> bool:
        self._cancel_after()
        for stream in self._streams():
            if stream.camera_reader is not None:
                stream.camera_reader.stop()
                stream.camera_reader = None
            if stream.cap is not None:
                stream.cap.release()
                stream.cap = None
            stream.preview_frame_id = 0

        left_cap = open_camera(self.left.camera_index)
        right_cap = open_camera(self.right.camera_index)
        if not left_cap.isOpened() or not right_cap.isOpened():
            if left_cap.isOpened():
                left_cap.release()
            if right_cap.isOpened():
                right_cap.release()
            messagebox.showerror(
                "Camera error",
                "Could not open both cameras. Check permissions and selections.",
            )
            self.status_var.set("One or both cameras unavailable.")
            return False

        self.left.cap = left_cap
        self.right.cap = right_cap
        self.record_fps = configure_camera_fps(self.left.cap)
        configure_camera_fps(self.right.cap)

        width = int(self.left.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.left.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.panel_size = panel_size_for_frame(width, height, STEREO_DISPLAY_MAX_SIZE)

        self.left.camera_reader = CameraReader(self.left.cap)
        self.right.camera_reader = CameraReader(self.right.cap)
        self.left.camera_reader.start()
        self.right.camera_reader.start()

        self.status_var.set(
            f"{self._camera_name(self.left.camera_index)} + "
            f"{self._camera_name(self.right.camera_index)} — live preview "
            f"@ {self.record_fps:.0f} fps. Press Start recording to save "
            f"{LEFT_VIDEO.name} and {RIGHT_VIDEO.name}"
        )
        self._schedule_record_preview()
        return True

    def _make_frame_consumer(self, stream: CameraStream):
        def on_frame(frame: np.ndarray) -> None:
            if stream.writer is not None:
                stream.writer.write(frame)

        return on_frame

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

        left_path = self.left.video_path
        right_path = self.right.video_path
        if not left_path.is_file() or not right_path.is_file():
            self.status_var.set(
                "No stereo pair loaded. Record one or use Open stereo pair…"
            )
            self.video_label.configure(image="", text="No videos loaded")
            return

        self.left.cap = cv2.VideoCapture(str(left_path))
        self.right.cap = cv2.VideoCapture(str(right_path))
        if not self._both_caps_open():
            messagebox.showerror(
                "Playback error",
                f"Could not open:\n{left_path}\n{right_path}",
            )
            self.status_var.set("Failed to open stereo pair.")
            return

        left_count = int(self.left.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        right_count = int(self.right.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if left_count and right_count:
            self.frame_count = min(left_count, right_count)
        else:
            self.frame_count = max(left_count, right_count)

        self.fps = self.left.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.left.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.left.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.panel_size = panel_size_for_frame(width, height, STEREO_DISPLAY_MAX_SIZE)
        self.frame_index = 0
        self._show_frame_at(0)
        self._update_status()

    def _schedule_record_preview(self) -> None:
        if self.mode.get() != "record":
            return
        if self.left.camera_reader is None or self.right.camera_reader is None:
            return

        left_ok, left_frame, left_id = self.left.camera_reader.get_latest_frame()
        right_ok, right_frame, right_id = self.right.camera_reader.get_latest_frame()
        if (
            left_ok
            and right_ok
            and left_frame is not None
            and right_frame is not None
            and (
                left_id != self.left.preview_frame_id
                or right_id != self.right.preview_frame_id
            )
        ):
            self.left.preview_frame_id = left_id
            self.right.preview_frame_id = right_id
            self.left.last_raw_frame = left_frame
            self.right.last_raw_frame = right_frame
            self._display_stereo_frames(left_frame, right_frame)

        delay = max(1, int(1000 / self.record_fps))
        self.after_id = self.root.after(delay, self._schedule_record_preview)

    def _toggle_recording(self) -> None:
        if not self._cameras_ready_for_record():
            return

        if not self.recording:
            left_reader = self.left.camera_reader
            right_reader = self.right.camera_reader
            assert left_reader is not None and right_reader is not None

            self.left.writer = create_writer(
                self.record_fps,
                left_reader.frame_width,
                left_reader.frame_height,
                LEFT_VIDEO,
            )
            self.right.writer = create_writer(
                self.record_fps,
                right_reader.frame_width,
                right_reader.frame_height,
                RIGHT_VIDEO,
            )
            if not self.left.writer.isOpened() or not self.right.writer.isOpened():
                messagebox.showerror(
                    "Recording error",
                    f"Could not create {LEFT_VIDEO.name} and {RIGHT_VIDEO.name}",
                )
                for stream in self._streams():
                    if stream.writer is not None:
                        stream.writer.release()
                        stream.writer = None
                return

            self.recording = True
            if self.left.camera_reader is not None:
                self.left.camera_reader.set_frame_consumer(self._make_frame_consumer(self.left))
            if self.right.camera_reader is not None:
                self.right.camera_reader.set_frame_consumer(
                    self._make_frame_consumer(self.right)
                )
            self.record_btn.configure(text="Stop recording")
            self._set_camera_controls_enabled(False)
            self.status_var.set(
                f"Recording {LEFT_VIDEO.name} + {RIGHT_VIDEO.name} "
                f"@ {self.record_fps:.0f} fps…"
            )
        else:
            self.recording = False
            for stream in self._streams():
                if stream.camera_reader is not None:
                    stream.camera_reader.set_frame_consumer(None)
                if stream.writer is not None:
                    stream.writer.release()
                    stream.writer = None
                stream.video_path = stream.default_video
            self.record_btn.configure(text="Start recording")
            self._set_camera_controls_enabled(True)
            self.status_var.set(
                f"Saved {LEFT_VIDEO.name} and {RIGHT_VIDEO.name}. "
                "Switch to Playback to review."
            )

    def _open_videos(self) -> None:
        left_path = filedialog.askopenfilename(
            title="Open left video",
            initialdir=RECORDINGS_DIR,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not left_path:
            return
        right_path = filedialog.askopenfilename(
            title="Open right video",
            initialdir=RECORDINGS_DIR,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not right_path:
            return
        self.left.video_path = Path(left_path)
        self.right.video_path = Path(right_path)
        self.mode.set("playback")
        self._enter_playback_mode()

    def _filtered_frame(
        self,
        stream: CameraStream,
        frame: np.ndarray,
        *,
        previous_frame: np.ndarray | None = None,
        window_frames: list[np.ndarray] | None = None,
        warmup_frames: list[np.ndarray] | None = None,
        video_fps: float | None = None,
    ) -> np.ndarray:
        return stream.frame_filter.apply(
            frame,
            previous_frame=previous_frame,
            window_frames=window_frames,
            warmup_frames=warmup_frames,
            video_fps=video_fps,
        )

    def _display_stereo_frames(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        *,
        left_previous: np.ndarray | None = None,
        right_previous: np.ndarray | None = None,
        left_window: list[np.ndarray] | None = None,
        right_window: list[np.ndarray] | None = None,
        left_warmup: list[np.ndarray] | None = None,
        right_warmup: list[np.ndarray] | None = None,
        video_fps: float | None = None,
    ) -> None:
        if self._is_stereo_tracking():
            left_filtered, right_filtered = self.stereo_tracking.apply(
                left_frame,
                right_frame,
                main_warmup_frames=left_warmup,
                main_previous_frame=left_previous,
                secondary_previous_frame=right_previous,
                video_fps=video_fps,
            )
        else:
            left_filtered = self._filtered_frame(
                self.left,
                left_frame,
                previous_frame=left_previous,
                window_frames=left_window,
                warmup_frames=left_warmup,
                video_fps=video_fps,
            )
            right_filtered = self._filtered_frame(
                self.right,
                right_frame,
                previous_frame=right_previous,
                window_frames=right_window,
                warmup_frames=right_warmup,
                video_fps=video_fps,
            )
        self.frame_photo = stereo_frame_to_photo(
            left_filtered, right_filtered, self.panel_size
        )
        self.video_label.configure(image=self.frame_photo, text="")

    def _show_frame_at(self, index: int) -> bool:
        if self.left.cap is None or self.right.cap is None:
            return False

        index = max(0, index)
        if self.frame_count > 0:
            index = min(index, self.frame_count - 1)

        ok_left, left_frame = read_frame_at(self.left.cap, index)
        ok_right, right_frame = read_frame_at(self.right.cap, index)
        if not ok_left or left_frame is None or not ok_right or right_frame is None:
            return False

        self.frame_index = index
        video_fps = self.fps

        if self._is_stereo_tracking():
            left_warmup = gru_warmup_frames_if_needed(
                self.left.cap,
                self.frame_index,
                self.left.gru_stream_frame_index,
                self.stereo_tracking.throw_buffer_size(),
            )
            right_previous = previous_frame_for_diff(self.right.cap, self.frame_index)
            left_previous = previous_frame_for_diff(self.left.cap, self.frame_index)
            left_window = None
            right_window = None
            right_warmup = None
        else:
            left_previous, left_window, left_warmup = filter_inputs_for_playback(
                self.left.cap,
                self.left.frame_filter,
                self.frame_index,
                self.left.gru_stream_frame_index,
            )
            right_previous, right_window, right_warmup = filter_inputs_for_playback(
                self.right.cap,
                self.right.frame_filter,
                self.frame_index,
                self.right.gru_stream_frame_index,
            )

        self._display_stereo_frames(
            left_frame,
            right_frame,
            left_previous=left_previous,
            right_previous=right_previous,
            left_window=left_window,
            right_window=right_window,
            left_warmup=left_warmup,
            right_warmup=right_warmup,
            video_fps=video_fps,
        )

        if self._is_stereo_tracking():
            self.left.gru_stream_frame_index = self.frame_index
        else:
            if uses_gru_streaming(self.left.frame_filter):
                self.left.gru_stream_frame_index = self.frame_index
            if uses_gru_streaming(self.right.frame_filter):
                self.right.gru_stream_frame_index = self.frame_index

        self._update_status()
        return True

    def _update_status(self) -> None:
        total = self.frame_count if self.frame_count > 0 else "?"
        left_name = self.left.video_path.name
        right_name = self.right.video_path.name
        time_s = self.frame_index / self.fps if self.fps else 0
        self.status_var.set(
            f"{left_name} + {right_name} — frame {self.frame_index + 1} / {total} "
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
        if not self._both_caps_open() or self.mode.get() != "playback":
            return
        if self.frame_count and self.frame_index >= self.frame_count - 1:
            self._show_frame_at(0)
        self.playing = True
        self._schedule_playback()

    def _pause(self) -> None:
        self.playing = False
        self._cancel_after()

    def _schedule_playback(self) -> None:
        if not self.playing or not self._both_caps_open():
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
