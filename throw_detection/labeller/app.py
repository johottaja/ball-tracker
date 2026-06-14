from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import ImageTk

from pose_detection import PoseDetector
from video_viewer.config import DISPLAY_MAX_SIZE
from video_viewer.display import fit_size, frame_to_photo

from throw_detection.dataset import (
    LabelingSession,
    dataset_path_for_set,
    merge_labels_from_file,
    save_dataset,
)
from throw_detection.labeller.clips import (
    clip_frame_count,
    extract_pose_from_video,
    list_clips,
    read_frame_at,
)
from throw_detection.labeller.overlay import render_labeller_frame


class ThrowLabellerApp:
    def __init__(self, root: tk.Tk, set_name: str) -> None:
        self.root = root
        self.set_name = set_name
        self.root.title(f"Throw Labeller — {set_name}")
        self.root.minsize(640, 520)

        clip_paths = list_clips(set_name)
        if not clip_paths:
            messagebox.showerror(
                "No clips",
                f"No clip_*.mp4 files found in recordings/{set_name}/",
            )
            root.after(0, root.destroy)
            return

        self.session = LabelingSession(set_name=set_name, clip_paths=clip_paths)
        self.detector = PoseDetector()
        self.dataset_path = dataset_path_for_set(set_name)

        self.cap: cv2.VideoCapture | None = None
        self.clip_index = 0
        self.frame_index = 0
        self.frame_count = 0
        self.fps = 30.0
        self.playing = False
        self.after_id: str | None = None
        self.frame_photo: ImageTk.PhotoImage | None = None
        self.display_size = fit_size(640, 480)

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.dataset_path.is_file():
            merge_labels_from_file(self.session, self.dataset_path)

        self._load_clip(self.clip_index)

    def _build_ui(self) -> None:
        self.video_label = ttk.Label(self.root, text="Loading…", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)

        row = ttk.Frame(controls)
        row.pack()
        for text, command in (
            ("|◀ Beginning", self._go_to_start),
            ("◀ Frame", self._step_backward),
            ("Play", self._play),
            ("Pause", self._pause),
            ("Frame ▶", self._step_forward),
        ):
            ttk.Button(row, text=text, command=command, takefocus=False).pack(
                side=tk.LEFT,
                padx=2,
            )

        clip_row = ttk.Frame(controls)
        clip_row.pack(pady=(8, 0))
        for text, command in (
            ("◀ Clip", self._previous_clip),
            ("Clip ▶", self._next_clip),
            ("Save", self._save),
        ):
            ttk.Button(clip_row, text=text, command=command, takefocus=False).pack(
                side=tk.LEFT,
                padx=2 if text != "Save" else 12,
            )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var).pack(
            fill=tk.X,
            padx=8,
            pady=(0, 8),
        )

    def _bind_keys(self) -> None:
        # bind_all so shortcuts work even when a widget had focus; return "break"
        # so Space does not activate the default button (|◀ Beginning → frame 0).
        bindings = {
            "<Left>": self._step_backward,
            "<Right>": self._step_forward,
            "<Up>": self._play,
            "<Down>": self._pause,
            "<space>": self._on_space,
        }
        for sequence, handler in bindings.items():
            self.root.bind_all(
                sequence,
                lambda _e, handler=handler: self._key_handler(handler),
            )
        self.root.focus_set()

    def _key_handler(self, handler) -> str:
        handler()
        return "break"

    def _current_clip_path(self) -> Path:
        return self.session.clip_paths[self.clip_index]

    def _current_labels(self) -> np.ndarray:
        return self.session.label_array_for_clip(
            self._current_clip_path(),
            self.frame_count,
        )

    def _cancel_after(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _release_capture(self) -> None:
        self._cancel_after()
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _load_clip(self, clip_index: int) -> None:
        self._pause()
        self._release_capture()

        self.clip_index = max(0, min(clip_index, len(self.session.clip_paths) - 1))
        clip_path = self._current_clip_path()

        self.cap = cv2.VideoCapture(str(clip_path))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open {clip_path}")
            return

        self.frame_count = clip_frame_count(self.cap)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_size = fit_size(width, height, DISPLAY_MAX_SIZE)

        resolved = clip_path.resolve()
        cached = self.session.pose_cache.get(resolved)
        if cached is not None:
            self.frame_count = len(cached.sides)

        self.session.label_array_for_clip(resolved, self.frame_count)
        self.frame_index = 0
        self._show_frame_at(0)

    def _read_frame_at(self, index: int) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        return read_frame_at(self.cap, index)

    def _show_frame_at(self, index: int) -> bool:
        if self.cap is None:
            return False

        index = max(0, index)
        if self.frame_count > 0:
            index = min(index, self.frame_count - 1)

        ok, frame = self._read_frame_at(index)
        if not ok or frame is None:
            return False

        self.frame_index = index
        label = int(self._current_labels()[self.frame_index])
        rendered = render_labeller_frame(frame, label, detector=self.detector)
        self.frame_photo = frame_to_photo(rendered, self.display_size)
        self.video_label.configure(image=self.frame_photo, text="")
        self._update_status()
        return True

    def _update_status(self) -> None:
        clip_path = self._current_clip_path()
        total = self.frame_count if self.frame_count > 0 else "?"
        label = int(self._current_labels()[self.frame_index]) if self.frame_count else 0
        clip_num = self.clip_index + 1
        clip_total = len(self.session.clip_paths)
        time_s = self.frame_index / self.fps if self.fps else 0
        self.status_var.set(
            f"{self.session.sanitized_set_name} — clip {clip_num}/{clip_total}: "
            f"{clip_path.name} — frame {self.frame_index + 1}/{total} "
            f"({time_s:.2f}s) — label {label} — save: {self.dataset_path.name}",
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
        if self.cap is None:
            return
        if self.frame_count and self.frame_index >= self.frame_count - 1:
            self._show_frame_at(0)
        self.playing = True
        self._schedule_playback()

    def _pause(self) -> None:
        self.playing = False
        self._cancel_after()

    def _schedule_playback(self) -> None:
        if not self.playing:
            return

        next_index = self.frame_index + 1
        if self.frame_count and next_index >= self.frame_count:
            self.playing = False
            self._show_frame_at(self.frame_count - 1)
            self.status_var.set(self.status_var.get() + " — end of clip")
            return

        if not self._show_frame_at(next_index):
            self.playing = False
            return

        delay = max(1, int(1000 / self.fps))
        self.after_id = self.root.after(delay, self._schedule_playback)

    def _on_space(self, _event: tk.Event | None = None) -> None:
        if self.frame_count == 0:
            return

        labels = self._current_labels()
        current = int(labels[self.frame_index])
        if current == 0:
            labels[self.frame_index] = 1
            self._step_forward()
        else:
            labels[self.frame_index] = 0
            self._show_frame_at(self.frame_index)

    def _previous_clip(self) -> None:
        if self.clip_index > 0:
            self._load_clip(self.clip_index - 1)

    def _next_clip(self) -> None:
        if self.clip_index < len(self.session.clip_paths) - 1:
            self._load_clip(self.clip_index + 1)

    def _ensure_pose_for_clip(self, clip_path: Path) -> None:
        resolved = clip_path.resolve()
        if resolved in self.session.pose_cache:
            return

        def on_progress(done: int, total: int) -> None:
            self.status_var.set(
                f"Extracting pose for {clip_path.name}… {done}/{total}",
            )
            self.root.update_idletasks()

        self.session.pose_cache[resolved] = extract_pose_from_video(
            clip_path,
            self.detector,
            progress=on_progress,
        )

    def _save(self) -> None:
        for clip_path in self.session.clip_paths:
            self._ensure_pose_for_clip(clip_path)
            self.session.label_array_for_clip(
                clip_path.resolve(),
                len(self.session.pose_cache[clip_path.resolve()].sides),
            )

        try:
            save_dataset(self.dataset_path, self.session)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        messagebox.showinfo(
            "Saved",
            f"Dataset written to\n{self.dataset_path}",
        )
        self._update_status()

    def _on_close(self) -> None:
        for sequence in ("<Left>", "<Right>", "<Up>", "<Down>", "<space>"):
            self.root.unbind_all(sequence)
        self._release_capture()
        self.root.destroy()
