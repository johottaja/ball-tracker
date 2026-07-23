from __future__ import annotations

import shutil
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
from video_viewer.ball_motion import (
    BALL_DETECTION_METHOD_LABELS,
    BallDetectionMethod,
    uses_mog2_component,
)
from video_viewer.config import THROW_MODEL_PATH
from video_viewer.playback import step_index_by_seconds
from video_viewer.playback_cache import PlaybackCache
from video_viewer.recording import create_writer
from video_viewer.stereo_playback import (
    StereoFrameReader,
    gru_warmup_for_timeline_playback,
    stereo_timeline_ball_mask_inputs,
)
from video_viewer.stereo_timeline import (
    STEREO_TIMELINE_FILENAME,
    finalize_stereo_recording,
    load_stereo_timeline_for_videos,
    stereo_timeline_path_for,
)
from video_viewer.yolo_batch import try_load_pose_cache
from video_viewer.gru_batch import try_load_gru_cache
from stereo_viewer.config import LEFT_VIDEO as STEREO_VIEWER_LEFT_VIDEO
from stereo_viewer.config import RIGHT_VIDEO as STEREO_VIEWER_RIGHT_VIDEO
from stereo_viewer.stereo_tracking import StereoTrackingProcessor

from calibration import (
    CameraLayoutDialog,
    TableCalibration,
    TableCalibrationDialog,
    capture_stereo_pair,
    load_calibration,
)

from .config import (
    DISPLAY_MAX_SIZE,
    GAMES_DIR,
    LEFT_VIDEO,
    RECORDINGS_DIR,
    RIGHT_VIDEO,
    TARGET_FPS,
    YOLO_INFERENCES,
    GRU_INFERENCES,
)
from .display import panel_size_for_frame, stereo_frame_to_photo
from .game_data import GameSession, load_game
from .paths import latest_game_json
from .process_dialog import ProcessGameDialog


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
    recorded_timestamps: list[float] = field(default_factory=list)
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
        GAMES_DIR.mkdir(parents=True, exist_ok=True)

        self.mode = tk.StringVar(value="record")
        self.left = CameraStream("Left", LEFT_VIDEO)
        self.right = CameraStream("Right", RIGHT_VIDEO)

        self.calibration = load_calibration()
        self.current_game_json: Path | None = None
        self.loaded_session: GameSession | None = None

        self.recording = False
        self.playing = False
        self.frame_index = 0
        self.frame_count = 0
        self.fps = TARGET_FPS
        self.record_fps = TARGET_FPS
        self.frame_photo = None
        self.after_id: str | None = None
        self.stereo_timeline = None
        self.stereo_reader: StereoFrameReader | None = None
        self.stereo_tracking = StereoTrackingProcessor(enable_framesync=False)
        self.stereo_tracking.set_calibration(self.calibration)
        self.playback_cache = PlaybackCache()
        self.ball_method_var = tk.StringVar(
            value=BALL_DETECTION_METHOD_LABELS[BallDetectionMethod.MOG2_CLOSING]
        )
        self.panel_size = panel_size_for_frame(640, 480, DISPLAY_MAX_SIZE)
        self.cameras: list[CameraDevice] = []
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

        ttk.Button(toolbar, text="Calibrate", command=self._open_calibration).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(toolbar, text="Camera layout", command=self._open_camera_layout).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(
            toolbar,
            text="Import from stereo viewer",
            command=self._import_from_stereo_viewer,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(toolbar, text="Process game…", command=self._open_process_dialog).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(toolbar, text="Open stereo pair…", command=self._open_videos).pack(
            side=tk.RIGHT
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
        ball_row = ttk.Frame(self.playback_controls)
        ball_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ball_row, text="Ball detection:").pack(side=tk.LEFT)
        self.ball_method_combo = ttk.Combobox(
            ball_row,
            textvariable=self.ball_method_var,
            state="readonly",
            width=34,
        )
        self.ball_method_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.ball_method_combo.configure(
            values=[BALL_DETECTION_METHOD_LABELS[m] for m in BallDetectionMethod]
        )
        self.ball_method_combo.bind("<<ComboboxSelected>>", self._on_ball_method_change)

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
        label = self.ball_method_var.get()
        for method, method_label in BALL_DETECTION_METHOD_LABELS.items():
            if method_label == label:
                return method
        return BallDetectionMethod.MOG2_CLOSING

    def _on_ball_method_change(self, _event: object | None = None) -> None:
        method = self._selected_ball_method()
        self.stereo_tracking.set_ball_detection_method(method)
        self.playback_cache.clear_filter_outputs()
        self.playback_cache.clear_motion_masks()
        for stream in self._streams():
            stream.gru_stream_frame_index = None
            stream.mog2_stream_frame_index = None
        if self.mode.get() == "playback":
            self._refresh_visible_frame()

    def _load_playback_caches(self) -> None:
        if self.frame_count <= 0:
            return
        if not try_load_pose_cache(
            YOLO_INFERENCES,
            self.frame_count,
            "stereo",
            self.playback_cache,
            timeline_signature=(
                self.stereo_timeline.signature if self.stereo_timeline is not None else None
            ),
        ):
            return
        if THROW_MODEL_PATH is not None and THROW_MODEL_PATH.is_file():
            try_load_gru_cache(
                GRU_INFERENCES,
                self.frame_count,
                THROW_MODEL_PATH,
                self.playback_cache,
                layout="stereo",
                timeline_signature=(
                    self.stereo_timeline.signature
                    if self.stereo_timeline is not None
                    else None
                ),
            )

    def _reset_playback_tracking(self) -> None:
        self.stereo_tracking.reset()
        self.playback_cache.clear_filter_outputs()
        self.playback_cache.clear_motion_masks()
        for stream in self._streams():
            stream.gru_stream_frame_index = None
            stream.mog2_stream_frame_index = None

    def _open_calibration(self) -> None:
        frames = capture_stereo_pair(
            mode=self.mode.get(),
            frame_index=self.frame_index,
            left_last_raw=self.left.last_raw_frame,
            right_last_raw=self.right.last_raw_frame,
            left_cap=self.left.cap,
            right_cap=self.right.cap,
        )
        if frames is None:
            messagebox.showwarning(
                "No frames",
                "Open cameras or a stereo video pair and show a frame before calibrating.",
                parent=self.root,
            )
            return
        left_frame, right_frame = frames
        TableCalibrationDialog(
            self.root,
            left_frame,
            right_frame,
            DISPLAY_MAX_SIZE,
            on_save=self._on_calibration_saved,
            frame_provider=lambda: capture_stereo_pair(
                mode=self.mode.get(),
                frame_index=self.frame_index,
                left_last_raw=self.left.last_raw_frame,
                right_last_raw=self.right.last_raw_frame,
                left_cap=self.left.cap,
                right_cap=self.right.cap,
            ),
        ).show()

    def _on_calibration_saved(self, calibration: TableCalibration) -> None:
        self.calibration = calibration
        self.stereo_tracking.set_calibration(calibration)

    def _open_camera_layout(self) -> None:
        CameraLayoutDialog(self.root, self.calibration).show()

    def _load_game_session(self, path: Path | None = None) -> None:
        if path is None:
            path = self.current_game_json or latest_game_json()
        self.current_game_json = path
        self.loaded_session = load_game(path) if path is not None else None

    def _on_game_processed(self, game_json_path: Path, _throws: int) -> None:
        self._load_game_session(game_json_path)
        if self.mode.get() == "playback" and self.stereo_reader is not None:
            self.playback_cache.clear()
            self._load_playback_caches()
            self._reset_playback_tracking()
            self._refresh_visible_frame()
        self._update_status_extra()

    def _stereo_video_metadata(self) -> tuple[float, int] | None:
        left_path = self.left.video_path or self.left.default_video
        right_path = self.right.video_path or self.right.default_video
        if not left_path.is_file() or not right_path.is_file():
            return None

        left_cap = cv2.VideoCapture(str(left_path))
        right_cap = cv2.VideoCapture(str(right_path))
        if not left_cap.isOpened() or not right_cap.isOpened():
            left_cap.release()
            right_cap.release()
            return None

        left_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        right_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = left_cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        left_cap.release()
        right_cap.release()

        timeline = load_stereo_timeline_for_videos(
            left_path,
            left_frame_count=left_count,
            right_frame_count=right_count,
            fps=fps,
        )
        frame_count = timeline.master_count
        if frame_count <= 0:
            return None
        return fps, frame_count

    def _open_process_dialog(
        self,
        *,
        fps: float | None = None,
        frame_count: int | None = None,
    ) -> None:
        if fps is None or frame_count is None:
            metadata = self._stereo_video_metadata()
            if metadata is None:
                messagebox.showwarning(
                    "No recording",
                    "Record a game or open a stereo video pair first.",
                    parent=self.root,
                )
                return
            file_fps, file_frame_count = metadata
            fps = fps if fps is not None else file_fps
            frame_count = frame_count if frame_count is not None else file_frame_count

        left_path = self.left.video_path or self.left.default_video
        right_path = self.right.video_path or self.right.default_video
        ProcessGameDialog(
            self.root,
            left_video=left_path,
            right_video=right_path,
            fps=fps,
            frame_count=frame_count,
            calibration=self.calibration,
            on_complete=self._on_game_processed,
        ).show()

    def _update_status_extra(self) -> None:
        base = self.status_var.get().split(" — throws:")[0]
        if self.loaded_session is None or not self.loaded_session.throws:
            return
        throws = len(self.loaded_session.throws)
        last_speed = self.loaded_session.throws[-1].speed_m_s
        extra = f" — throws: {throws}"
        if last_speed is not None:
            extra += f", last speed: {last_speed:.1f} m/s"
        game_name = (
            self.current_game_json.name
            if self.current_game_json is not None
            else "game"
        )
        extra += f" ({game_name})"
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
        if stream.camera_reader is not None:
            stream.camera_reader.set_frame_consumer(None)
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
        self.stereo_reader = None
        self.stereo_timeline = None
        self.playback_cache.clear()
        self.stereo_tracking.reset()
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

        self.status_var.set(
            f"{self._camera_name(self.left.camera_index)} + "
            f"{self._camera_name(self.right.camera_index)} — live preview "
            f"@ {self.record_fps:.0f} fps. Press Start recording to save "
            f"{LEFT_VIDEO.name} and {RIGHT_VIDEO.name}"
        )
        self._schedule_record_preview()
        return True

    def _make_frame_consumer(self, stream: CameraStream):
        def on_frame(frame: np.ndarray, captured_at: float) -> None:
            if stream.writer is not None:
                stream.writer.write(frame)
                stream.recorded_frame_count += 1
                stream.recorded_timestamps.append(captured_at)

        return on_frame

    def _finalize_stereo_recording(self) -> tuple[int, str]:
        left_count = self.left.recorded_frame_count
        right_count = self.right.recorded_frame_count
        if (
            left_count <= 0
            or right_count <= 0
            or not self.left.recorded_timestamps
            or not self.right.recorded_timestamps
        ):
            return 0, ""

        timeline = finalize_stereo_recording(
            left_timestamps=self.left.recorded_timestamps,
            right_timestamps=self.right.recorded_timestamps,
            fps=self.record_fps,
            left_video=self.left.default_video,
        )
        detail = (
            f"{timeline.master_count} paired master slots "
            f"(left {left_count}, right {right_count})"
        )
        return timeline.master_count, detail

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
        self.fps = self.left.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.stereo_timeline = load_stereo_timeline_for_videos(
            left_path,
            left_frame_count=left_count,
            right_frame_count=right_count,
            fps=self.fps,
        )
        self.stereo_reader = StereoFrameReader(
            self.left.cap,
            self.right.cap,
            self.stereo_timeline,
        )
        self.frame_count = self.stereo_timeline.master_count
        width = int(self.left.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.left.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.panel_size = panel_size_for_frame(width, height, DISPLAY_MAX_SIZE)
        self.frame_index = 0
        self._load_game_session()
        method = self._selected_ball_method()
        self.stereo_tracking.set_ball_detection_method(method)
        self.playback_cache.clear()
        self._load_playback_caches()
        self._reset_playback_tracking()

        self._show_frame_at(0)
        self._update_status()

    def _display_stereo_frames(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        *,
        frame_index: int,
        left_previous: np.ndarray | None = None,
        right_previous: np.ndarray | None = None,
        left_next: np.ndarray | None = None,
        right_next: np.ndarray | None = None,
        left_mog2_warmup: list[np.ndarray] | None = None,
        right_mog2_warmup: list[np.ndarray] | None = None,
        left_warmup: list[np.ndarray] | None = None,
        left_warmup_start_index: int | None = None,
    ) -> None:
        ball_method = self._selected_ball_method()
        left_filtered, right_filtered = self.stereo_tracking.apply(
            left_frame,
            right_frame,
            frame_index=frame_index,
            main_warmup_frames=left_warmup,
            main_warmup_start_index=left_warmup_start_index,
            main_previous_frame=left_previous,
            main_next_frame=left_next,
            main_mog2_warmup_frames=left_mog2_warmup,
            secondary_previous_frame=right_previous,
            secondary_next_frame=right_next,
            secondary_mog2_warmup_frames=right_mog2_warmup,
            video_fps=self.fps,
            cache=self.playback_cache,
            ball_method=ball_method,
            stereo_timeline=(
                self.stereo_reader.timeline
                if self.stereo_reader is not None
                else None
            ),
        )
        self.frame_photo = stereo_frame_to_photo(
            left_filtered, right_filtered, self.panel_size
        )
        self.video_label.configure(image=self.frame_photo, text="")

    def _show_raw_stereo(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> None:
        self.frame_photo = stereo_frame_to_photo(
            left_frame, right_frame, self.panel_size
        )
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

            if YOLO_INFERENCES.is_file():
                YOLO_INFERENCES.unlink()
            if GRU_INFERENCES.is_file():
                GRU_INFERENCES.unlink()

            for stream in self._streams():
                stream.recorded_frame_count = 0
                stream.recorded_timestamps = []

            self.recording = True
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

            frame_count, timeline_detail = self._finalize_stereo_recording()

            for stream in self._streams():
                stream.video_path = stream.default_video
                stream.recorded_frame_count = 0
                stream.recorded_timestamps = []
            self.record_btn.configure(text="Start recording")
            self._set_camera_controls_enabled(True)

            saved = f"Saved {LEFT_VIDEO.name} and {RIGHT_VIDEO.name}"
            if frame_count > 0:
                saved += f" ({timeline_detail})"
            self.status_var.set(saved)
            self._open_process_dialog(fps=self.record_fps, frame_count=frame_count)

    def _import_from_stereo_viewer(self) -> None:
        if self.recording:
            messagebox.showwarning(
                "Recording in progress",
                "Stop recording before importing from stereo viewer.",
                parent=self.root,
            )
            return

        if (
            not STEREO_VIEWER_LEFT_VIDEO.is_file()
            or not STEREO_VIEWER_RIGHT_VIDEO.is_file()
        ):
            messagebox.showerror(
                "No stereo viewer recording",
                "stereo_viewer/recordings/left.mp4 and right.mp4 were not found.\n"
                "Record a stereo pair in stereo viewer first.",
                parent=self.root,
            )
            return

        if LEFT_VIDEO.is_file() or RIGHT_VIDEO.is_file():
            if not messagebox.askyesno(
                "Overwrite game tracker recordings?",
                "This replaces game_tracker/recordings/left.mp4 and right.mp4 "
                "with the current stereo viewer recordings.",
                parent=self.root,
            ):
                return

        self._release_capture()

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STEREO_VIEWER_LEFT_VIDEO, LEFT_VIDEO)
        shutil.copy2(STEREO_VIEWER_RIGHT_VIDEO, RIGHT_VIDEO)
        if YOLO_INFERENCES.is_file():
            YOLO_INFERENCES.unlink()
        if GRU_INFERENCES.is_file():
            GRU_INFERENCES.unlink()
        stereo_timeline = stereo_timeline_path_for(STEREO_VIEWER_LEFT_VIDEO)
        if stereo_timeline.is_file():
            shutil.copy2(stereo_timeline, RECORDINGS_DIR / STEREO_TIMELINE_FILENAME)

        self.left.video_path = LEFT_VIDEO
        self.right.video_path = RIGHT_VIDEO
        self.mode.set("playback")
        self._enter_playback_mode()

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
        if self.stereo_reader is None:
            return False

        index = max(0, index)
        if self.frame_count > 0:
            index = min(index, self.frame_count - 1)

        left_frame, right_frame = self.stereo_reader.read_at_master(index)
        if left_frame is None or right_frame is None:
            return False

        self.frame_index = index
        method = self._selected_ball_method()
        left_warmup, left_warmup_start_index = gru_warmup_for_timeline_playback(
            self.stereo_reader,
            self.frame_index,
            self.left.gru_stream_frame_index,
            self.stereo_tracking.throw_buffer_size(),
            self.playback_cache.main,
        )
        (
            left_previous,
            right_previous,
            left_next,
            right_next,
            left_mog2_warmup,
            right_mog2_warmup,
        ) = stereo_timeline_ball_mask_inputs(
            self.stereo_reader,
            method,
            self.frame_index,
            self.left.mog2_stream_frame_index,
            self.right.mog2_stream_frame_index,
            self.playback_cache.main,
            self.playback_cache.secondary,
        )
        self._display_stereo_frames(
            left_frame,
            right_frame,
            frame_index=self.frame_index,
            left_previous=left_previous,
            right_previous=right_previous,
            left_next=left_next,
            right_next=right_next,
            left_mog2_warmup=left_mog2_warmup,
            right_mog2_warmup=right_mog2_warmup,
            left_warmup=left_warmup,
            left_warmup_start_index=left_warmup_start_index,
        )
        self.left.gru_stream_frame_index = self.frame_index
        if uses_mog2_component(method):
            self.left.mog2_stream_frame_index = self.frame_index
            self.right.mog2_stream_frame_index = self.frame_index
        self._update_status()
        return True

    def _update_status(self) -> None:
        total = self.frame_count if self.frame_count > 0 else "?"
        left_name = self.left.video_path.name if self.left.video_path else "?"
        right_name = self.right.video_path.name if self.right.video_path else "?"
        if self.stereo_timeline is not None and self.stereo_timeline.master_count:
            time_s = self.stereo_timeline.master_times[self.frame_index]
        else:
            time_s = self.frame_index / self.fps if self.fps else 0
        hold_bits: list[str] = []
        if self.stereo_timeline is not None:
            if not self.stereo_timeline.captured_timestamps:
                hold_bits.append("synthetic alignment; processing disabled")
            if self.stereo_timeline.is_hold("left", self.frame_index):
                hold_bits.append("left hold")
            if self.stereo_timeline.is_hold("right", self.frame_index):
                hold_bits.append("right hold")
        hold_suffix = f" [{', '.join(hold_bits)}]" if hold_bits else ""
        self.status_var.set(
            f"{left_name} + {right_name} — frame {self.frame_index + 1} / {total} "
            f"({time_s:.2f}s @ {self.fps:.1f} fps){hold_suffix}"
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

        delay_ms = 33
        if self.stereo_timeline is not None:
            delay_ms = self.stereo_timeline.slot_duration_ms(self.frame_index)
        self.after_id = self.root.after(delay_ms, self._schedule_playback)

    def _on_close(self) -> None:
        self._release_capture()
        self.root.destroy()
