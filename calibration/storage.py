from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import CALIBRATION_JSON
from .layout import compute_calibration_layout
from .types import TableCalibration


def attach_layout_stats(calibration: TableCalibration) -> TableCalibration:
    # Layout statistics are derived entirely from the projection matrices. Always
    # regenerate them so calibration files saved by an earlier layout algorithm do
    # not retain stale camera pose or FOV diagnostics.
    return replace(calibration, layout=compute_calibration_layout(calibration))


def save_calibration(
    calibration: TableCalibration,
    path: Path = CALIBRATION_JSON,
) -> TableCalibration:
    calibration = attach_layout_stats(calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(calibration.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return calibration


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
        return attach_layout_stats(calibration)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None
