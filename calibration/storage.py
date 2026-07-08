from __future__ import annotations

import json
from pathlib import Path

from .config import CALIBRATION_JSON
from .types import TableCalibration


def save_calibration(
    calibration: TableCalibration,
    path: Path = CALIBRATION_JSON,
) -> None:
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
        return calibration
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None
