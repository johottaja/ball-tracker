from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import CALIBRATION_JSON
from .layout import compute_calibration_layout
from .types import TableCalibration


def attach_layout_stats(calibration: TableCalibration) -> TableCalibration:
    layout = calibration.layout
    needs_layout = (
        layout is None
        or not layout.cameras
        or (
            layout.stereo is None
            and calibration.camera("left") is not None
            and calibration.camera("right") is not None
        )
    )
    if not needs_layout:
        return calibration
    return replace(calibration, layout=compute_calibration_layout(calibration))


def save_calibration(
    calibration: TableCalibration,
    path: Path = CALIBRATION_JSON,
) -> None:
    calibration = attach_layout_stats(calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(calibration.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_calibration(path: Path = CALIBRATION_JSON) -> TableCalibration | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        calibration = TableCalibration.from_dict(data)
        if len(calibration.cameras) < 2:
            return None
        if calibration.layout is None or not calibration.layout.cameras:
            calibration = attach_layout_stats(calibration)
            save_calibration(calibration, path)
        return calibration
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None
