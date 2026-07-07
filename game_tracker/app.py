from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from video_viewer.ball_motion import (
    BALL_DETECTION_METHOD_LABELS,
    BallDetectionMethod,
    uses_mog2_component,
)
from video_viewer.camera import (
    CameraDevice,
    CameraReader,
    configure_camera_fps,
    open_camera,
    probe_cameras,
)
from video_viewer.playback import (
    gru_warmup_for_playback,
    stereo_ball_mask_playback_inputs,
    read_frame_at,
    step_index_by_seconds,
)
from video_viewer.playback_cache import PlaybackCache
from video_viewer.recording import create_writer, extend_video_evenly

from .config import (
    DISPLAY_MAX_SIZE,
    GAME_JSON,
    LEFT_VIDEO,
    RECORDINGS_DIR,
    RIGHT_VIDEO,
    TARGET_FPS,
)
from .display import panel_size_for_frame, stereo_frame_to_photo
from .game_data import load_game
from .processor import GameTrackingProcessor
from .setup_config import CameraSetup, load_setup_config, save_setup_config
from .setup_dialog import CameraSetupDialog


@dataclass
class CameraStream:
    label: str
    default_video: Path
    camera_index: int = 0
    cap: cv2.VideoCapture | None = None
    camera_reader: CameraReader | None = None
    writer: cv2.VideoWriter | None = None
    gru_stream_frame_index: int | None = None
    mog2_stream_frame_index: int | None = None
    last_raw_frame: np.ndarray | None = None
    recorded_frame_count: int = 0
    preview_frame_id: int = 0
    video_path: Path | None = None

    def __post_init__(self) -> None:
        if self.video_path is None:
            self.video_path = self.default_video


class GameTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Beer Pong Game Tracker")
        self.root.minsize(960, 520)

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self.mode = tk.StringVar(value="record")
        self.left = CameraStream("Left", LEFT_VIDEO)
        self.right = CameraStream("Right", RIGHT_VIDEO)

        self.camera_setup = load_setup_config()
        self.processor = GameTrackingProcessor()
        self.processor.set_camera_setup(self.camera_setup)
        self.processor.set_on_throw_recorded(self._on_throw_recorded)
        self.playback_cache = PlaybackCache()

        self.recording = False
        self.playing = False
        self.frame_index = 0
        self.frame_count = 0
        self.fps = TARGET_FPS
        self.record_fps = TARGET_FPS
        self.frame_photo = None
        self.after_id: str | None = None
        self.panel_size = panel_size_for_frame(640, 480, DISPLAY_MAX_SIZE)
        self.cameras: list[CameraDevice] = []
        self._setup_dialog = CameraSetupDialog(
            root,
            self.camera_setup,
            on_save=self._on_camera_setup_saved,
        )

        self._build_ui()
        self._bind_playback_keys()
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

        ttk.Button(toolbar, text="Camera setup…", command=self._open_setup_dialog).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(toolbar, text="Open stereo pair…", command=self._open_videos).pack(
            side=tk.RIGHT
        )

        controls = ttk.Frame(self.root, padding=(8, 4, 8, 0))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Ball detection:").pack(side=tk.LEFT)
        self.ball_method_var = tk.StringVar(value=BallDetectionMethod.MOG2_CLOSING.value)
        self.ball_method_combo = ttk.Combobox(
            controls,
            values=[BALL_DETECTION_METHOD_LABELS[m] for m in BallDetectionMethod],
            state="readonly",
            width=28,
        )
        self.ball_method_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.ball_method_combo.set(
            BALL_DETECTION_METHOD_LABELS[BallDetectionMethod.MOG2_CLOSING]
        )
        self.ball_method_combo.bind("<<ComboboxSelected>>", self._on_ball_method_change)

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
        step_back_btn = ttk.Button(btn_row, text="◀ Frame")
        step_back_btn.pack(side=tk.LEFT, padx=2)
        step_back_btn.bind("<Button-1>", self._on_step_backward_click)
        ttk.Button(btn_row, text="Play", command=self._play).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="Pause", command=self._pause).pack(side=tk.LEFT, padx=2)
        step_forward_btn = ttk.Button(btn_row, text="Frame ▶")
        step_forward_btn.pack(side=tk.LEFT, padx=2)
        step_forward_btn.bind("<Button-1>", self._on_step_forward_click)

    def _streams(self) -> tuple[CameraStream, CameraStream]:
        return self.left, self.right

    def _selected_ball_method(self) -> BallDetectionMethod:
        selected = self.ball_method_combo.get()
        for method, label in BALL_DETECTION_METHOD_LABELS.items():
            if label == selected:
                return method
        return BallDetectionMethod(self.ball_method_var.get())

    def _on_ball_method_change(self, _event: object | None = None) -> None:
        method = self._selected_ball_method()
        self.ball_method_var.set(method.value)
        self.processor.set_ball_detection_method(self._selected_ball_method())
        self.playback_cache.clear_motion_masks()
        for stream in self._streams():
            stream.gru_stream_frame_index = None
            stream.mog2_stream_frame_index = None
        self._refresh_visible_frame()

    def _open_setup_dialog(self) -> None:
        self._setup_dialog = CameraSetupDialog(
            self.root,
            self.camera_setup,
            on_save=self._on_camera_setup_saved,
        )
        self._setup_dialog.show()

    def _on_camera_setup_saved(self, setup: CameraSetup) -> None:
        self.camera_setup = setup
        self.processor.set_camera_setup(setup)

    def _on_throw_recorded(self, _throw) -> None:
        self._update_status_extra()

    def _update_status_extra(self) -> None:
        base = self.status_var.get().split(" — throws:")[0]
        throws = self.processor.state.throw_count
        speed = self.processor.state.last_speed_m_s
        extra = f" — throws: {throws}"
        if speed is not None:
            extra += f", last speed: {speed:.1f} m/s"
        self.status_var.set(base + extra)

    def _refresh_visible_frame(self) -> None:
        if self.mode.get() == "playback":
            if self._both_caps_open():
                self._show_frame_at(self.frame_index)
        elif self.left.last_raw_frame is not None and self.right.last_raw_frame is not None:
            self._show_raw_stereo(self.left.last_raw_frame, self.right.last_raw_frame)

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
        stream.mog2_stream_frame_index = None
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
        stream.recorded_frame_count = 0

    def _release_capture(self) -> None:
        self._cancel_after()
        self.playing = False
        self.processor.reset_tracking()
        self.playback_cache.clear()
        for stream in self._streams():
            self._release_stream(stream)

    def _set_camera_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.left_camera_combo.configure(state=state)
        self.right_camera_combo.configure(state=state)
        self.refresh_cameras_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _camera_name(self, index: int) -> str:
        for camera in self.cameras:
            if camera.index == index:
                return camera.label
        return f"Camera {index}"

    def _refresh_cameras(self) -> None:
        self.cameras = probe_cameras()
        if len(self.cameras) < 2:
            self.status_var.set(
                "Need at least two cameras. Connect devices and click Refresh."
            )
            self.video_label.configure(image="", text="No cameras found")
            return

        labels = [camera.label for camera in self.cameras]
        self.left_camera_combo.configure(values=labels)
        self.right_camera_combo.configure(values=labels)

        if self.left.camera_index not in [c.index for c in self.cameras]:
            self.left.camera_index = self.cameras[0].index
        if self.right.camera_index not in [c.index for c in self.cameras]:
            self.right.camera_index = self.cameras[1].index
        if self.left.camera_index == self.right.camera_index:
            self.right.camera_index = self.cameras[1].index

        self.left_camera_var.set(self._camera_name(self.left.camera_index))
        self.right_camera_var.set(self._camera_name(self.right.camera_index))
        self._open_cameras()

    def _resolve_camera_index(self, label: str) -> int | None:
        for camera in self.cameras:
            if camera.label == label:
                return camera.index
        return None

    def _on_left_camera_selected(self, _event: object | None = None) -> None:
        index = self._resolve_camera_index(self.left_camera_var.get())
        if index is None:
            return
        if index == self.right.camera_index:
            messagebox.showwarning(
                "Camera conflict",
                "Left and right cameras must be different.",
            )
            self.left_camera_var.set(self._camera_name(self.left.camera_index))
            return
        self.left.camera_index = index
        self._open_cameras()

    def _on_right_camera_selected(self, _event: object | None = None) -> None:
        index = self._resolve_camera_index(self.right_camera_var.get())
        if index is None:
            return
        if index == self.left.camera_index:
            messagebox.showwarning(
                "Camera conflict",
                "Left and right cameras must be different.",
            )
            self.right_camera_var.set(self._camera_name(self.right.camera_index))
            return
        self.right.camera_index = index
        self._open_cameras()

    def _open_cameras(self) -> bool:
        self._release_capture()

        left_cap = open_camera(self.left.camera_index)
        right_cap = open_camera(self.right.camera_index)
        if not left_cap.isOpened() or not right_cap.isOpened():
            messagebox.showerror(
                "Camera error",
                "Could not open one or both cameras.",
            )
            return False

        self.record_fps = configure_camera_fps(left_cap)
        configure_camera_fps(right_cap)

        ok_left, left_frame = left_cap.read()
        ok_right, right_frame = right_cap.read()
        if not ok_left or left_frame is None or not ok_right or right_frame is None:
            left_cap.release()
            right_cap.release()
            messagebox.showerror("Camera error", "Could not read frames from cameras.")
            return False

        self.panel_size = panel_size_for_frame(
            left_frame.shape[1],
            left_frame.shape[0],
            DISPLAY_MAX_SIZE,
        )

        self.left.cap = left_cap
        self.right.cap = right_cap
        self.left.camera_reader = CameraReader(left_cap)
        self.right.camera_reader = CameraReader(right_cap)
        self.left.camera_reader.start()
        self.right.camera_reader.start()
        self.left.last_raw_frame = left_frame
        self.right.last_raw_frame = right_frame
        self.processor.reset_tracking()

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
                stream.recorded_frame_count += 1

        return on_frame

    def _equalize_stereo_recordings(self) -> tuple[int, list[str]]:
        target_count = max(
            self.left.recorded_frame_count,
            self.right.recorded_frame_count,
        )
        extended_labels: list[str] = []
        for stream, label in (
            (self.left, "left"),
            (self.right, "right"),
        ):
            source_count = stream.recorded_frame_count
            if source_count <= 0 or source_count >= target_count:
                continue
            extend_video_evenly(
                stream.default_video,
                source_count=source_count,
                target_count=target_count,
                fps=self.record_fps,
            )
            extended_labels.append(f"{label} +{target_count - source_count}")
        return target_count, extended_labels

    def _begin_game_session(self, frame_count: int) -> None:
        self.processor.begin_session(
            fps=self.record_fps,
            frame_count=frame_count,
        )

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
        assert left_path is not None and right_path is not None
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
        self.panel_size = panel_size_for_frame(width, height, DISPLAY_MAX_SIZE)
        self.frame_index = 0
        self.playback_cache.clear()

        existing = load_game(GAME_JSON)
        if existing is not None:
            self.processor.load_session(existing)
        else:
            self.processor.begin_session(
                fps=self.fps,
                frame_count=self.frame_count,
            )

        self._show_frame_at(0)
        self._update_status()

    def _playback_warmup_inputs(
        self, frame_index: int
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        list[np.ndarray] | None,
        list[np.ndarray] | None,
        list[np.ndarray] | None,
        int | None,
    ]:
        left_warmup, left_warmup_start_index = gru_warmup_for_playback(
            self.left.cap,
            frame_index,
            self.left.gru_stream_frame_index,
            self.processor.throw_buffer_size(),
            self.playback_cache.main,
        )
        method = self._selected_ball_method()
        (
            left_previous,
            right_previous,
            left_mog2_warmup,
            right_mog2_warmup,
        ) = stereo_ball_mask_playback_inputs(
            self.left.cap,
            self.right.cap,
            method,
            frame_index,
            self.left.mog2_stream_frame_index,
            self.right.mog2_stream_frame_index,
            self.playback_cache.main,
            self.playback_cache.secondary,
        )
        return (
            left_previous,
            right_previous,
            left_mog2_warmup,
            right_mog2_warmup,
            left_warmup,
            left_warmup_start_index,
        )

    def _show_raw_stereo(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> None:
        self.frame_photo = stereo_frame_to_photo(
            left_frame, right_frame, self.panel_size
        )
        self.video_label.configure(image=self.frame_photo, text="")

    def _display_stereo_frames(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        *,
        frame_index: int = 0,
        left_previous: np.ndarray | None = None,
        right_previous: np.ndarray | None = None,
        left_mog2_warmup: list[np.ndarray] | None = None,
        right_mog2_warmup: list[np.ndarray] | None = None,
        left_warmup: list[np.ndarray] | None = None,
        left_warmup_start_index: int | None = None,
        video_fps: float | None = None,
    ) -> None:
        if self.mode.get() == "record":
            self._show_raw_stereo(left_frame, right_frame)
            return

        left_out, right_out = self.processor.apply(
            left_frame,
            right_frame,
            frame_index=frame_index,
            main_warmup_frames=left_warmup,
            main_warmup_start_index=left_warmup_start_index,
            main_previous_frame=left_previous,
            main_mog2_warmup_frames=left_mog2_warmup,
            secondary_previous_frame=right_previous,
            secondary_mog2_warmup_frames=right_mog2_warmup,
            video_fps=video_fps,
            cache=self.playback_cache,
        )
        self.frame_photo = stereo_frame_to_photo(left_out, right_out, self.panel_size)
        self.video_label.configure(image=self.frame_photo, text="")

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
            self._show_raw_stereo(left_frame, right_frame)

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

            for stream in self._streams():
                stream.recorded_frame_count = 0

            self.recording = True
            self.processor.reset_tracking()
            if self.left.camera_reader is not None:
                self.left.camera_reader.set_frame_consumer(
                    self._make_frame_consumer(self.left)
                )
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

            for stream in self._streams():
                if stream.writer is not None:
                    stream.writer.release()
                    stream.writer = None

            frame_count, extended_labels = self._equalize_stereo_recordings()

            for stream in self._streams():
                stream.video_path = stream.default_video
                stream.recorded_frame_count = 0
            self.record_btn.configure(text="Start recording")
            self._set_camera_controls_enabled(True)

            self._begin_game_session(frame_count)
            self.processor.reset_tracking()

            saved = f"Saved {LEFT_VIDEO.name} and {RIGHT_VIDEO.name}"
            if frame_count > 0:
                saved += f" ({frame_count} frames each"
                if extended_labels:
                    saved += f", extended {', '.join(extended_labels)}"
                saved += ")"
            self.status_var.set(
                f"{saved}. Switch to Playback to analyze throws."
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

        (
            left_previous,
            right_previous,
            left_mog2_warmup,
            right_mog2_warmup,
            left_warmup,
            left_warmup_start_index,
        ) = self._playback_warmup_inputs(index)

        self._display_stereo_frames(
            left_frame,
            right_frame,
            frame_index=self.frame_index,
            left_previous=left_previous,
            right_previous=right_previous,
            left_mog2_warmup=left_mog2_warmup,
            right_mog2_warmup=right_mog2_warmup,
            left_warmup=left_warmup,
            left_warmup_start_index=left_warmup_start_index,
            video_fps=video_fps,
        )

        self.left.gru_stream_frame_index = self.frame_index
        if uses_mog2_component(self._selected_ball_method()):
            self.left.mog2_stream_frame_index = self.frame_index
            self.right.mog2_stream_frame_index = self.frame_index

        self._update_status()
        return True

    def _update_status(self) -> None:
        total = self.frame_count if self.frame_count > 0 else "?"
        left_name = self.left.video_path.name if self.left.video_path else "?"
        right_name = self.right.video_path.name if self.right.video_path else "?"
        time_s = self.frame_index / self.fps if self.fps else 0
        self.status_var.set(
            f"{left_name} + {right_name} — frame {self.frame_index + 1} / {total} "
            f"({time_s:.2f}s @ {self.fps:.1f} fps)"
        )
        self._update_status_extra()

    def _bind_playback_keys(self) -> None:
        bindings = {
            "<Left>": self._step_backward,
            "<Right>": self._step_forward,
            "<Shift-Left>": self._skip_backward,
            "<Shift-Right>": self._skip_forward,
            "<Up>": self._play,
            "<Down>": self._pause,
        }
        for sequence, handler in bindings.items():
            self.root.bind_all(
                sequence,
                lambda _e, handler=handler: self._playback_key_handler(handler),
            )
        self.root.focus_set()

    def _playback_key_handler(self, handler) -> str:
        handler()
        return "break"

    def _on_step_backward_click(self, event: tk.Event) -> str:
        if event.state & 0x1:
            self._skip_backward()
        else:
            self._step_backward()
        return "break"

    def _on_step_forward_click(self, event: tk.Event) -> str:
        if event.state & 0x1:
            self._skip_forward()
        else:
            self._step_forward()
        return "break"

    def _go_to_start(self) -> None:
        self._pause()
        self._show_frame_at(0)

    def _step_backward(self) -> None:
        self._pause()
        self._show_frame_at(self.frame_index - 1)

    def _step_forward(self) -> None:
        self._pause()
        self._show_frame_at(self.frame_index + 1)

    def _skip_backward(self) -> None:
        self._pause()
        self._show_frame_at(
            step_index_by_seconds(
                self.frame_index,
                self.fps,
                1.0,
                forward=False,
                frame_count=self.frame_count,
            )
        )

    def _skip_forward(self) -> None:
        self._pause()
        self._show_frame_at(
            step_index_by_seconds(
                self.frame_index,
                self.fps,
                1.0,
                forward=True,
                frame_count=self.frame_count,
            )
        )

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
        save_setup_config(self.camera_setup)
        self._release_capture()
        self.root.destroy()
