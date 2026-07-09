from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from video_viewer.config import THROW_MODEL_PATH
from video_viewer.gru_batch import (
    GruBatchProgress,
    gru_cache_status,
    gru_cache_status_label,
    populate_mono_gru_cache,
    populate_stereo_gru_cache,
    run_mono_gru_inference_phase,
    run_stereo_gru_inference_phase,
)
from video_viewer.playback_cache import PlaybackCache
from video_viewer.yolo_batch import (
    YoloBatchProgress,
    YoloCacheLayout,
    YoloCacheStatus,
    load_mono_yolo_inferences,
    load_stereo_yolo_inferences,
    populate_mono_pose_cache,
    populate_stereo_pose_cache,
    run_mono_yolo_inference_phase,
    run_stereo_yolo_inference_phase,
    yolo_cache_status,
    yolo_cache_status_label,
)


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "—"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass
class _ProgressUpdate:
    phase: str
    progress: YoloBatchProgress | GruBatchProgress


@dataclass
class _RunDone:
    error: str | None


class PoseEstimationPanel:
    """Playback-only controls for optional batched YOLO pose + GRU preprocessing."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Tk,
        *,
        layout: YoloCacheLayout,
        cache_path: Path,
        gru_cache_path: Path | None = None,
        include_gru: bool = False,
        run_button_text: str = "Run pose estimation",
        on_busy_change: Callable[[bool], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self._root = root
        self._layout = layout
        self._cache_path = cache_path
        self._gru_cache_path = gru_cache_path
        self._include_gru = include_gru and gru_cache_path is not None
        self._run_button_text = run_button_text
        self._on_busy_change = on_busy_change
        self._on_complete = on_complete

        self._playback_cache: PlaybackCache | None = None
        self._frame_count = 0
        self._mono_video: Path | None = None
        self._left_video: Path | None = None
        self._right_video: Path | None = None
        self._fps = 30.0

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._yolo_disk_status: YoloCacheStatus = "missing"
        self._gru_disk_status = gru_cache_status(
            gru_cache_path or Path(),
            0,
            THROW_MODEL_PATH,
        ) if self._include_gru else "no_model"
        self._cached_frame_count = 0

        self._frame = ttk.Frame(parent)
        self._frame.pack(fill=tk.X, pady=(8, 0))

        row = ttk.Frame(self._frame)
        row.pack(fill=tk.X)

        self._run_btn = ttk.Button(
            row,
            text=run_button_text,
            command=self._start,
        )
        self._run_btn.pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value=self._default_status_text())
        ttk.Label(row, textvariable=self._status_var).pack(side=tk.LEFT, padx=(12, 0))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress = ttk.Progressbar(
            self._frame,
            variable=self._progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self._detail_var = tk.StringVar(value="")
        ttk.Label(self._frame, textvariable=self._detail_var).pack(anchor=tk.W, pady=(2, 0))

        self._poll_queue()

    @property
    def is_running(self) -> bool:
        return self._running

    def set_playback_cache(self, cache: PlaybackCache) -> None:
        self._playback_cache = cache

    def set_frame_count(self, frame_count: int) -> None:
        self._frame_count = frame_count
        self._refresh_disk_status()

    def set_mono_video(self, path: Path) -> None:
        self._mono_video = path

    def set_stereo_videos(self, left: Path, right: Path, *, fps: float) -> None:
        self._left_video = left
        self._right_video = right
        self._fps = fps

    def _default_status_text(self) -> str:
        if self._include_gru:
            return "Preprocess: not run"
        return "Pose cache: not run"

    def status_suffix(self) -> str:
        if self._running:
            return "Preprocess: running…" if self._include_gru else "Pose cache: inferring…"
        if self._include_gru:
            return self._preprocess_status_suffix()
        return yolo_cache_status_label(
            self._yolo_disk_status,
            cached_frames=self._cached_frame_count,
            expected_frames=self._frame_count,
        )

    def _preprocess_status_suffix(self) -> str:
        yolo_label = yolo_cache_status_label(
            self._yolo_disk_status,
            cached_frames=self._cached_frame_count,
            expected_frames=self._frame_count,
        ).removeprefix("Pose cache: ")
        gru_label = gru_cache_status_label(
            self._gru_disk_status,
            cached_frames=self._cached_frame_count,
            expected_frames=self._frame_count,
        ).removeprefix("GRU cache: ")
        if self._yolo_disk_status == "ready" and self._gru_disk_status == "ready":
            return f"Preprocess: ready ({self._cached_frame_count:,} frames)"
        return f"Preprocess: pose {yolo_label.lower()} — GRU {gru_label.lower()}"

    def try_load_cached_poses(self) -> bool:
        if self._playback_cache is None or self._frame_count <= 0:
            return False
        from video_viewer.yolo_batch import try_load_pose_cache

        loaded = try_load_pose_cache(
            self._cache_path,
            self._frame_count,
            self._layout,
            self._playback_cache,
        )
        if loaded and self._include_gru and self._gru_cache_path is not None:
            from video_viewer.gru_batch import try_load_gru_cache

            try_load_gru_cache(
                self._gru_cache_path,
                self._frame_count,
                THROW_MODEL_PATH,
                self._playback_cache,
                layout=self._layout,
            )
        return loaded

    def on_enter_playback(self) -> None:
        self._refresh_disk_status()
        self.try_load_cached_poses()

    def on_leave_playback(self) -> None:
        if self._running:
            self._cancel_event.set()

    def set_playback_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled and not self._running else tk.DISABLED
        self._run_btn.configure(state=state)

    def _refresh_disk_status(self) -> None:
        self._yolo_disk_status = yolo_cache_status(
            self._cache_path,
            self._frame_count,
            self._layout,
        )
        self._cached_frame_count = 0
        if self._yolo_disk_status in ("ready", "stale"):
            try:
                import numpy as np

                with np.load(self._cache_path, allow_pickle=False) as data:
                    self._cached_frame_count = int(data["frame_count"])
            except (OSError, KeyError, TypeError, ValueError):
                self._yolo_disk_status = "missing"

        if self._include_gru and self._gru_cache_path is not None:
            self._gru_disk_status = gru_cache_status(
                self._gru_cache_path,
                self._frame_count,
                THROW_MODEL_PATH,
            )
        else:
            self._gru_disk_status = "no_model"

        self._status_var.set(self.status_suffix())
        if not self._running:
            self._progress.pack_forget()
            self._detail_var.set("")

    def _set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._progress.pack(fill=tk.X, pady=(4, 0))
            self._progress_var.set(0.0)
            self._run_btn.configure(text="Running…", state=tk.DISABLED)
        else:
            self._progress.pack_forget()
            self._run_btn.configure(text=self._run_button_text)
            self.set_playback_controls_enabled(True)
        if self._on_busy_change is not None:
            self._on_busy_change(running)

    def _start(self) -> None:
        if self._running or self._frame_count <= 0:
            return
        if self._layout == "mono":
            if self._mono_video is None or not self._mono_video.is_file():
                messagebox.showerror(
                    "Preprocess" if self._include_gru else "Pose estimation",
                    "No video loaded.",
                    parent=self._root,
                )
                return
        else:
            if (
                self._left_video is None
                or self._right_video is None
                or not self._left_video.is_file()
                or not self._right_video.is_file()
            ):
                messagebox.showerror(
                    "Preprocess" if self._include_gru else "Pose estimation",
                    "No stereo pair loaded.",
                    parent=self._root,
                )
                return

        if self._include_gru and THROW_MODEL_PATH is None:
            messagebox.showerror(
                "Preprocess",
                "No GRU model in throw_detection/models/.",
                parent=self._root,
            )
            return

        cache_ready = self._yolo_disk_status == "ready"
        if self._include_gru:
            cache_ready = cache_ready and self._gru_disk_status == "ready"

        overwrite = False
        if cache_ready:
            title = "Re-run preprocess?" if self._include_gru else "Re-run pose estimation?"
            prompt = (
                "Matching preprocess caches already exist. Re-run and overwrite them?"
                if self._include_gru
                else "A matching pose cache already exists. Re-run and overwrite it?"
            )
            if not messagebox.askyesno(title, prompt, parent=self._root):
                return
            overwrite = True

        need_yolo = self._yolo_disk_status != "ready" or overwrite
        need_gru = self._include_gru and (
            self._gru_disk_status != "ready" or overwrite
        )

        self._cancel_event.clear()
        self._set_running(True)
        self._status_var.set(
            "Preprocess: running…" if self._include_gru else "Pose cache: inferring…"
        )
        self._detail_var.set("Starting…")

        def run() -> None:
            try:
                def on_yolo_progress(update: YoloBatchProgress) -> None:
                    self._event_queue.put(
                        _ProgressUpdate(phase="yolo", progress=update)
                    )

                def on_gru_progress(update: GruBatchProgress) -> None:
                    self._event_queue.put(
                        _ProgressUpdate(phase="gru", progress=update)
                    )

                if self._layout == "mono":
                    assert self._mono_video is not None
                    if need_yolo:
                        yolo_store = run_mono_yolo_inference_phase(
                            video_path=self._mono_video,
                            output_path=self._cache_path,
                            frame_count=self._frame_count,
                            progress=on_yolo_progress,
                            cancel_check=self._cancel_event.is_set,
                        )
                    else:
                        yolo_store = load_mono_yolo_inferences(self._cache_path)
                    if self._playback_cache is not None and not self._cancel_event.is_set():
                        populate_mono_pose_cache(yolo_store, self._playback_cache.main)
                    if (
                        need_gru
                        and self._gru_cache_path is not None
                        and THROW_MODEL_PATH is not None
                        and not self._cancel_event.is_set()
                    ):
                        gru_store = run_mono_gru_inference_phase(
                            yolo_store=yolo_store,
                            model_path=THROW_MODEL_PATH,
                            output_path=self._gru_cache_path,
                            progress=on_gru_progress,
                            cancel_check=self._cancel_event.is_set,
                        )
                        if self._playback_cache is not None:
                            populate_mono_gru_cache(gru_store, self._playback_cache.main)
                else:
                    assert self._left_video is not None and self._right_video is not None
                    if need_yolo:
                        yolo_store = run_stereo_yolo_inference_phase(
                            left_path=self._left_video,
                            right_path=self._right_video,
                            output_path=self._cache_path,
                            frame_count=self._frame_count,
                            fps=self._fps,
                            progress=on_yolo_progress,
                            cancel_check=self._cancel_event.is_set,
                        )
                    else:
                        yolo_store = load_stereo_yolo_inferences(self._cache_path)
                    if self._playback_cache is not None and not self._cancel_event.is_set():
                        populate_stereo_pose_cache(yolo_store, self._playback_cache)
                    if (
                        need_gru
                        and self._gru_cache_path is not None
                        and THROW_MODEL_PATH is not None
                        and not self._cancel_event.is_set()
                    ):
                        gru_store = run_stereo_gru_inference_phase(
                            yolo_store=yolo_store,
                            model_path=THROW_MODEL_PATH,
                            output_path=self._gru_cache_path,
                            progress=on_gru_progress,
                            cancel_check=self._cancel_event.is_set,
                        )
                        if self._playback_cache is not None:
                            populate_stereo_gru_cache(gru_store, self._playback_cache)

                self._event_queue.put(_RunDone(error=None))
            except Exception as exc:
                self._event_queue.put(_RunDone(error=str(exc)))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._event_queue.get_nowait()
                if isinstance(item, _ProgressUpdate):
                    self._on_progress(item)
                elif isinstance(item, _RunDone):
                    self._on_finished(item)
        except queue.Empty:
            pass
        self._root.after(100, self._poll_queue)

    def _on_progress(self, item: _ProgressUpdate) -> None:
        progress = item.progress
        self._progress_var.set(progress.fraction * 100.0)
        current = progress.frame_index + 1
        phase_label = "YOLO pose" if item.phase == "yolo" else "GRU throw"
        detail = f"{phase_label} — frame {current:,} / {progress.frame_count:,}"
        eta = progress.eta_s
        if eta is not None:
            detail += (
                f" — ETA {_format_duration(eta)}"
                f" (elapsed {_format_duration(progress.elapsed_s)})"
            )
        else:
            detail += f" — elapsed {_format_duration(progress.elapsed_s)}"
        self._detail_var.set(detail)

    def _on_finished(self, item: _RunDone) -> None:
        self._worker = None
        self._set_running(False)

        dialog_title = "Preprocess failed" if self._include_gru else "Pose estimation failed"
        if item.error is not None:
            self._refresh_disk_status()
            messagebox.showerror(dialog_title, item.error, parent=self._root)
            return

        self._refresh_disk_status()
        done_label = "Preprocessed" if self._include_gru else "Inferred"
        self._detail_var.set(f"Done — {self._frame_count:,} frames {done_label.lower()}")
        if self._on_complete is not None:
            self._on_complete()
