from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from calibration import TableCalibration
from video_viewer.ball_motion import BALL_DETECTION_METHOD_LABELS, BallDetectionMethod
from video_viewer.config import THROW_MODEL_PATH

from .batch_process import BatchProgress, process_game_recording
from .paths import default_game_json_name, game_json_path


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "—"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _ball_method_from_label(label: str) -> BallDetectionMethod:
    for method, method_label in BALL_DETECTION_METHOD_LABELS.items():
        if method_label == label:
            return method
    return BallDetectionMethod.MOG2_CLOSING


@dataclass
class _ProgressUpdate:
    progress: BatchProgress


@dataclass
class _ProcessDone:
    game_json_path: Path | None
    throws: int
    error: str | None


class _PhaseProgressUi:
    def __init__(self, parent: ttk.Frame, title: str) -> None:
        self.title_var = tk.StringVar(value=title)
        ttk.Label(parent, textvariable=self.title_var).pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            parent,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        ).pack(fill=tk.X, pady=(2, 4))
        self.detail_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.detail_var).pack(anchor=tk.W)


class ProcessGameDialog:
    """Post-recording dialog to name a game, pick ball detection, and process offline."""

    def __init__(
        self,
        parent: tk.Tk,
        *,
        left_video: Path,
        right_video: Path,
        fps: float,
        frame_count: int,
        calibration: TableCalibration | None,
        default_ball_method: BallDetectionMethod = BallDetectionMethod.MOG2_CLOSING,
        on_complete: Callable[[Path, int], None] | None = None,
    ) -> None:
        self._parent = parent
        self._left_video = left_video
        self._right_video = right_video
        self._fps = fps
        self._frame_count = frame_count
        self._calibration = calibration
        self._default_ball_method = default_ball_method
        self._on_complete = on_complete

        self._window: tk.Toplevel | None = None
        self._process_thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._event_queue: queue.Queue = queue.Queue()
        self._processing = False
        self._yolo_ui: _PhaseProgressUi | None = None
        self._gru_ui: _PhaseProgressUi | None = None
        self._tracking_ui: _PhaseProgressUi | None = None

    def show(self) -> None:
        if self._window is not None and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return

        win = tk.Toplevel(self._parent)
        win.title("Process game")
        win.transient(self._parent)
        win.grab_set()
        win.resizable(False, False)
        self._window = win

        root = ttk.Frame(win, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text=(
                f"Recording saved ({self._frame_count} frames @ {self._fps:.0f} fps). "
                "Name the game and choose ball detection before processing."
            ),
            wraplength=420,
        ).pack(anchor=tk.W, pady=(0, 12))

        name_row = ttk.Frame(root)
        name_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(name_row, text="Game file:").pack(side=tk.LEFT)
        self._name_var = tk.StringVar(value=default_game_json_name())
        self._name_entry = ttk.Entry(name_row, textvariable=self._name_var, width=36)
        self._name_entry.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)

        method_row = ttk.Frame(root)
        method_row.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(method_row, text="Ball detection:").pack(side=tk.LEFT)
        self._method_combo = ttk.Combobox(
            method_row,
            values=[BALL_DETECTION_METHOD_LABELS[m] for m in BallDetectionMethod],
            state="readonly",
            width=28,
        )
        self._method_combo.pack(side=tk.LEFT, padx=(8, 0))
        self._method_combo.set(BALL_DETECTION_METHOD_LABELS[self._default_ball_method])

        self._progress_frame = ttk.Frame(root)
        self._yolo_ui = _PhaseProgressUi(self._progress_frame, "YOLO pose inference")
        ttk.Separator(self._progress_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=8
        )
        self._gru_ui = _PhaseProgressUi(self._progress_frame, "GRU throw inference")
        ttk.Separator(self._progress_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=8
        )
        self._tracking_ui = _PhaseProgressUi(self._progress_frame, "Game tracking")
        self._summary_var = tk.StringVar(value="")
        ttk.Label(
            self._progress_frame,
            textvariable=self._summary_var,
        ).pack(anchor=tk.W, pady=(8, 0))

        btn_row = ttk.Frame(root)
        btn_row.pack(fill=tk.X, pady=(12, 0))
        self._process_btn = ttk.Button(
            btn_row,
            text="Process game",
            command=self._start_processing,
        )
        self._process_btn.pack(side=tk.LEFT)
        self._close_btn = ttk.Button(btn_row, text="Close", command=self._close)
        self._close_btn.pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_queue()

    def _set_processing_ui(self, active: bool) -> None:
        self._processing = active
        state = tk.DISABLED if active else tk.NORMAL
        self._name_entry.configure(state=state)
        self._method_combo.configure(state="readonly" if not active else tk.DISABLED)
        self._process_btn.configure(state=state)
        if active:
            self._progress_frame.pack(fill=tk.X, pady=(0, 4))
            if self._yolo_ui is not None:
                self._yolo_ui.progress_var.set(0.0)
                self._yolo_ui.detail_var.set("Waiting…")
            if self._gru_ui is not None:
                self._gru_ui.progress_var.set(0.0)
                self._gru_ui.detail_var.set("Waiting…")
            if self._tracking_ui is not None:
                self._tracking_ui.progress_var.set(0.0)
                self._tracking_ui.detail_var.set("Waiting…")
            self._summary_var.set("")
            self._process_btn.configure(text="Processing…")
        else:
            self._process_btn.configure(text="Process game")

    def _start_processing(self) -> None:
        if self._processing:
            return

        output_path = game_json_path(self._name_var.get())
        if output_path.exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"{output_path.name} already exists. Overwrite?",
                parent=self._window,
            ):
                return

        self._cancel_event.clear()
        self._set_processing_ui(True)

        ball_method = _ball_method_from_label(self._method_combo.get())

        def run() -> None:
            try:
                def on_progress(update: BatchProgress) -> None:
                    self._event_queue.put(_ProgressUpdate(progress=update))

                processor = process_game_recording(
                    left_path=self._left_video,
                    right_path=self._right_video,
                    game_json_path=output_path,
                    ball_method=ball_method,
                    calibration=self._calibration,
                    fps=self._fps,
                    frame_count=self._frame_count,
                    progress=on_progress,
                    cancel_check=self._cancel_event.is_set,
                )
                self._event_queue.put(
                    _ProcessDone(
                        game_json_path=output_path,
                        throws=processor.state.throw_count,
                        error=None,
                    )
                )
            except Exception as exc:
                self._event_queue.put(
                    _ProcessDone(
                        game_json_path=None,
                        throws=0,
                        error=str(exc),
                    )
                )

        self._process_thread = threading.Thread(target=run, daemon=True)
        self._process_thread.start()

    def _poll_queue(self) -> None:
        if self._window is None or not self._window.winfo_exists():
            return

        try:
            while True:
                item = self._event_queue.get_nowait()
                if isinstance(item, _ProgressUpdate):
                    self._on_progress(item.progress)
                elif isinstance(item, _ProcessDone):
                    self._on_finished(item)
        except queue.Empty:
            pass

        self._window.after(100, self._poll_queue)

    def _phase_ui(self, progress: BatchProgress) -> _PhaseProgressUi | None:
        if progress.phase == "yolo":
            return self._yolo_ui
        if progress.phase == "gru":
            return self._gru_ui
        return self._tracking_ui

    def _on_progress(self, progress: BatchProgress) -> None:
        ui = self._phase_ui(progress)
        if ui is None:
            return

        ui.title_var.set(progress.phase_title)
        ui.progress_var.set(progress.fraction * 100.0)
        current = progress.frame_index + 1
        detail = f"Frame {current:,} / {progress.frame_count:,}"
        if progress.phase == "tracking":
            detail += (
                f" — {progress.throws} throw"
                f"{'s' if progress.throws != 1 else ''} detected"
            )
        eta = progress.eta_s
        if eta is not None:
            detail += (
                f" — ETA {_format_duration(eta)}"
                f" (elapsed {_format_duration(progress.elapsed_s)})"
            )
        else:
            detail += f" — elapsed {_format_duration(progress.elapsed_s)}"
        ui.detail_var.set(detail)

    def _on_finished(self, item: _ProcessDone) -> None:
        self._set_processing_ui(False)
        self._process_thread = None

        if item.error is not None:
            self._summary_var.set("Processing failed.")
            messagebox.showerror("Processing failed", item.error, parent=self._window)
            return

        assert item.game_json_path is not None
        throws = item.throws

        if self._yolo_ui is not None:
            self._yolo_ui.progress_var.set(100.0)
            self._yolo_ui.detail_var.set(
                f"Done — {self._frame_count:,} frames inferred"
            )
        if self._gru_ui is not None:
            self._gru_ui.progress_var.set(100.0)
            if THROW_MODEL_PATH is None or not THROW_MODEL_PATH.is_file():
                self._gru_ui.detail_var.set("Skipped — no GRU model")
            elif self._gru_ui.detail_var.get() == "Waiting…":
                self._gru_ui.detail_var.set(
                    f"Done — {self._frame_count:,} frames from cache"
                )
            else:
                self._gru_ui.detail_var.set(
                    f"Done — {self._frame_count:,} frames inferred"
                )
        if self._tracking_ui is not None:
            self._tracking_ui.progress_var.set(100.0)
            self._tracking_ui.detail_var.set(
                f"Done — {throws} throw{'s' if throws != 1 else ''} saved"
            )
        self._summary_var.set(f"Saved to {item.game_json_path}")

        if self._on_complete is not None:
            self._on_complete(item.game_json_path, throws)

        messagebox.showinfo(
            "Game processed",
            f"Saved {throws} throw{'s' if throws != 1 else ''} to\n{item.game_json_path}",
            parent=self._window,
        )

    def _close(self) -> None:
        if self._processing and self._process_thread is not None:
            if not messagebox.askyesno(
                "Processing in progress",
                "Stop processing and close?",
                parent=self._window,
            ):
                return
            self._cancel_event.set()
            self._process_thread.join(timeout=2.0)

        if self._window is not None:
            self._window.grab_release()
            self._window.destroy()
            self._window = None
