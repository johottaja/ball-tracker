from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from training_recorder.paths import sanitize_training_set_name

from throw_detection.config import MODELS_DIR
from throw_detection.model import save_throw_model
from throw_detection.train import (
    TrainingHyperparameters,
    TrainingResult,
    hyperparameters_to_dict,
    list_training_sets,
    train_throw_model,
)


@dataclass
class _ProgressUpdate:
    epoch: int
    total_epochs: int
    message: str


@dataclass
class _TrainingDone:
    result: TrainingResult | None
    error: str | None


class ThrowTrainerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Throw Detection Trainer")
        self.root.minsize(720, 560)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.training_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.event_queue: queue.Queue = queue.Queue()
        self.training_result: TrainingResult | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_training_sets()
        self._poll_queue()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        set_row = ttk.Frame(main)
        set_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(set_row, text="Training set:").pack(side=tk.LEFT)
        self.set_var = tk.StringVar()
        self.set_combo = ttk.Combobox(
            set_row,
            textvariable=self.set_var,
            state="readonly",
            width=36,
        )
        self.set_combo.pack(side=tk.LEFT, padx=(4, 4), fill=tk.X, expand=True)
        ttk.Button(set_row, text="Refresh", command=self._refresh_training_sets).pack(
            side=tk.LEFT,
        )

        params = ttk.LabelFrame(main, text="Hyperparameters", padding=8)
        params.pack(fill=tk.X, pady=(0, 8))

        self.param_vars: dict[str, tk.Variable] = {
            "hidden_size": tk.IntVar(value=64),
            "num_layers": tk.IntVar(value=2),
            "dropout": tk.DoubleVar(value=0.1),
            "learning_rate": tk.DoubleVar(value=1e-3),
            "batch_size": tk.IntVar(value=64),
            "epochs": tk.IntVar(value=30),
            "validation_split": tk.DoubleVar(value=0.2),
            "seed": tk.IntVar(value=42),
            "pos_weight": tk.DoubleVar(value=1.0),
        }

        fields = (
            ("Hidden size", "hidden_size"),
            ("GRU layers", "num_layers"),
            ("Dropout", "dropout"),
            ("Learning rate", "learning_rate"),
            ("Batch size", "batch_size"),
            ("Epochs", "epochs"),
            ("Validation split", "validation_split"),
            ("Random seed", "seed"),
            ("Positive class weight", "pos_weight"),
        )

        for row_index, (label, key) in enumerate(fields):
            col = (row_index % 3) * 2
            row = row_index // 3
            ttk.Label(params, text=f"{label}:").grid(
                row=row,
                column=col,
                sticky=tk.W,
                padx=(0, 4),
                pady=2,
            )
            ttk.Entry(params, textvariable=self.param_vars[key], width=10).grid(
                row=row,
                column=col + 1,
                sticky=tk.W,
                padx=(0, 16),
                pady=2,
            )

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(0, 8))
        self.train_btn = ttk.Button(controls, text="Train", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(
            controls,
            text="Stop",
            command=self._stop_training,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            main,
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(fill=tk.X, pady=(0, 4))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main, textvariable=self.status_var).pack(anchor=tk.W, pady=(0, 4))

        self.log = scrolledtext.ScrolledText(main, height=14, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        save_frame = ttk.LabelFrame(main, text="Save trained model", padding=8)
        save_frame.pack(fill=tk.X)
        save_row = ttk.Frame(save_frame)
        save_row.pack(fill=tk.X)
        ttk.Label(save_row, text="Model name:").pack(side=tk.LEFT)
        self.model_name_var = tk.StringVar()
        self.model_name_entry = ttk.Entry(
            save_row,
            textvariable=self.model_name_var,
            width=32,
            state=tk.DISABLED,
        )
        self.model_name_entry.pack(side=tk.LEFT, padx=(4, 8), fill=tk.X, expand=True)
        self.save_btn = ttk.Button(
            save_row,
            text="Save",
            command=self._save_model,
            state=tk.DISABLED,
        )
        self.save_btn.pack(side=tk.LEFT)

    def _refresh_training_sets(self) -> None:
        sets = list_training_sets()
        self.set_combo["values"] = sets
        if sets and self.set_var.get() not in sets:
            self.set_var.set(sets[0])
        elif not sets:
            self.set_var.set("")
            self.status_var.set("No .npz files in throw_detection/training_sets/.")

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_training_active(self, active: bool) -> None:
        state = tk.DISABLED if active else tk.NORMAL
        self.train_btn.configure(state=state)
        self.stop_btn.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.set_combo.configure(state="disabled" if active else "readonly")
        for child in self.root.winfo_children():
            self._set_widget_state_recursive(child, active, skip_save=True)

    def _set_widget_state_recursive(
        self,
        widget: tk.Widget,
        training_active: bool,
        *,
        skip_save: bool,
    ) -> None:
        if isinstance(widget, ttk.Entry) and widget is not self.model_name_entry:
            widget.configure(state=tk.DISABLED if training_active else tk.NORMAL)
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, training_active, skip_save=skip_save)

    def _read_hyperparameters(self) -> TrainingHyperparameters:
        return TrainingHyperparameters(
            hidden_size=int(self.param_vars["hidden_size"].get()),
            num_layers=int(self.param_vars["num_layers"].get()),
            dropout=float(self.param_vars["dropout"].get()),
            learning_rate=float(self.param_vars["learning_rate"].get()),
            batch_size=int(self.param_vars["batch_size"].get()),
            epochs=int(self.param_vars["epochs"].get()),
            validation_split=float(self.param_vars["validation_split"].get()),
            seed=int(self.param_vars["seed"].get()),
            pos_weight=float(self.param_vars["pos_weight"].get()),
        )

    def _start_training(self) -> None:
        set_name = self.set_var.get().strip()
        if not set_name:
            messagebox.showerror("No training set", "Select a training set first.")
            return

        try:
            hyperparameters = self._read_hyperparameters()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        if hyperparameters.epochs < 1:
            messagebox.showerror("Invalid parameters", "Epochs must be at least 1.")
            return
        if not 0.0 <= hyperparameters.validation_split < 1.0:
            messagebox.showerror(
                "Invalid parameters",
                "Validation split must be between 0 and 1.",
            )
            return

        self.training_result = None
        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self._append_log(f"Training on {set_name}.npz …")
        self.status_var.set("Training…")
        self._set_training_active(True)
        self._disable_save()

        def run() -> None:
            try:
                def on_progress(**kwargs) -> None:
                    self.event_queue.put(
                        _ProgressUpdate(
                            epoch=kwargs["epoch"],
                            total_epochs=kwargs["total_epochs"],
                            message=kwargs["message"],
                        ),
                    )

                result = train_throw_model(
                    set_name,
                    hyperparameters,
                    progress=on_progress,
                    stop_event=self.stop_event,
                )
                self.event_queue.put(_TrainingDone(result=result, error=None))
            except Exception as exc:
                self.event_queue.put(_TrainingDone(result=None, error=str(exc)))

        self.training_thread = threading.Thread(target=run, daemon=True)
        self.training_thread.start()

    def _stop_training(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping after current epoch…")

    def _disable_save(self) -> None:
        self.model_name_entry.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        self.model_name_var.set("")

    def _enable_save(self, set_name: str) -> None:
        default_name = f"{set_name}_gru"
        self.model_name_var.set(default_name)
        self.model_name_entry.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.event_queue.get_nowait()
                if isinstance(item, _ProgressUpdate):
                    fraction = item.epoch / max(item.total_epochs, 1)
                    self.progress_var.set(fraction * 100.0)
                    self.status_var.set(item.message)
                    self._append_log(item.message)
                elif isinstance(item, _TrainingDone):
                    self._on_training_finished(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_training_finished(self, item: _TrainingDone) -> None:
        self._set_training_active(False)
        self.training_thread = None

        if item.error is not None:
            self.status_var.set("Training failed.")
            self._append_log(f"Error: {item.error}")
            messagebox.showerror("Training failed", item.error)
            return

        assert item.result is not None
        self.training_result = item.result
        stopped = self.stop_event.is_set()
        self.progress_var.set(100.0 if not stopped else self.progress_var.get())

        if stopped:
            self.status_var.set("Training stopped.")
            self._append_log("Training stopped by user.")
        else:
            self.status_var.set("Training complete.")
            self._append_log("Training complete.")

        metrics = item.result.final_metrics
        if metrics:
            summary = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            self._append_log(f"Final metrics: {summary}")

        self._enable_save(item.result.set_name)

    def _model_path_for_name(self, name: str) -> Path:
        sanitized = sanitize_training_set_name(name)
        return MODELS_DIR / f"{sanitized}.pt"

    def _save_model(self) -> None:
        if self.training_result is None:
            messagebox.showerror("Nothing to save", "Train a model first.")
            return

        name = self.model_name_var.get().strip()
        if not name:
            messagebox.showerror("No name", "Enter a model name.")
            return

        path = self._model_path_for_name(name)
        if path.exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"{path.name} already exists. Overwrite?",
            ):
                return

        try:
            save_throw_model(
                path,
                self.training_result.model,
                set_name=self.training_result.set_name,
                buffer_size=self.training_result.buffer_size,
                training_config=hyperparameters_to_dict(
                    self.training_result.hyperparameters,
                ),
                metrics={
                    **self.training_result.final_metrics,
                    "history": self.training_result.history,
                },
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        messagebox.showinfo("Saved", f"Model written to\n{path}")
        self.status_var.set(f"Saved {path.name}")

    def _on_close(self) -> None:
        if self.training_thread is not None and self.training_thread.is_alive():
            if not messagebox.askyesno(
                "Training in progress",
                "Training is still running. Stop and quit?",
            ):
                return
            self.stop_event.set()
            self.training_thread.join(timeout=2.0)
        self.root.destroy()
