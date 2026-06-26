from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from .ball_motion import BALL_DETECTION_METHOD_LABELS, BallDetectionMethod
from .filters import FILTER_LABELS, STEREO_ONLY_FILTER_IDS, FilterId


class FilterControls:
    """Filter combobox, ball-detection method, and Filters menu."""

    def __init__(
        self,
        root: tk.Misc,
        *,
        on_change: Callable[[], None],
        on_ball_method_change: Callable[[], None],
        combo_parent: tk.Misc | None = None,
        combo_width: int = 42,
        include_stereo: bool = False,
    ) -> None:
        self.root = root
        self.on_change = on_change
        self.on_ball_method_change = on_ball_method_change
        self._filter_ids = [
            filter_id
            for filter_id in FilterId
            if include_stereo or filter_id not in STEREO_ONLY_FILTER_IDS
        ]
        self.filter_var = tk.StringVar(value=FilterId.NONE.value)
        self._filter_value_by_label = {
            FILTER_LABELS[fid]: fid.value for fid in self._filter_ids
        }
        self._filter_label_by_value = {
            value: label for label, value in self._filter_value_by_label.items()
        }

        self.ball_method_var = tk.StringVar(value=BallDetectionMethod.MOG2_CLOSING.value)
        self._ball_method_value_by_label = {
            label: method.value
            for method, label in BALL_DETECTION_METHOD_LABELS.items()
        }
        self._ball_method_label_by_value = {
            value: label for label, value in self._ball_method_value_by_label.items()
        }

        parent = combo_parent if combo_parent is not None else root
        filter_row = ttk.Frame(parent, padding=(8, 0, 8, 0))
        filter_row.pack(fill=tk.X)
        ttk.Label(filter_row, text="Filter:").pack(side=tk.LEFT)
        self.filter_combo = ttk.Combobox(
            filter_row,
            state="readonly",
            width=combo_width,
        )
        self.filter_combo.pack(side=tk.LEFT, padx=(4, 16))
        self.filter_combo.configure(values=list(self._filter_value_by_label))
        self.filter_combo.set(FILTER_LABELS[FilterId.NONE])
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_combo)

        ttk.Label(filter_row, text="Ball detection:").pack(side=tk.LEFT)
        self.ball_method_combo = ttk.Combobox(
            filter_row,
            state="readonly",
            width=34,
        )
        self.ball_method_combo.pack(side=tk.LEFT, padx=(4, 0))
        self.ball_method_combo.configure(values=list(self._ball_method_value_by_label))
        self.ball_method_combo.set(
            BALL_DETECTION_METHOD_LABELS[BallDetectionMethod.MOG2_CLOSING]
        )
        self.ball_method_combo.bind("<<ComboboxSelected>>", self._on_ball_method_combo)

    def add_menu(self, menubar: tk.Menu) -> None:
        filters_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filters", menu=filters_menu)
        for filter_id in self._filter_ids:
            filters_menu.add_radiobutton(
                label=FILTER_LABELS[filter_id],
                variable=self.filter_var,
                value=filter_id.value,
                command=self.on_change,
            )

    def selected_filter_id(self) -> FilterId:
        return FilterId(self.filter_var.get())

    def selected_ball_detection_method(self) -> BallDetectionMethod:
        return BallDetectionMethod(self.ball_method_var.get())

    def sync_combo_from_var(self) -> None:
        label = self._filter_label_by_value.get(self.filter_var.get(), "")
        if label and self.filter_combo.get() != label:
            self.filter_combo.set(label)

        method_label = self._ball_method_label_by_value.get(
            self.ball_method_var.get(), ""
        )
        if method_label and self.ball_method_combo.get() != method_label:
            self.ball_method_combo.set(method_label)

    def _on_filter_combo(self, _event: object | None = None) -> None:
        label = self.filter_combo.get()
        value = self._filter_value_by_label.get(label)
        if value is not None:
            self.filter_var.set(value)
            self.on_change()

    def _on_ball_method_combo(self, _event: object | None = None) -> None:
        label = self.ball_method_combo.get()
        value = self._ball_method_value_by_label.get(label)
        if value is not None:
            self.ball_method_var.set(value)
            self.on_ball_method_change()
