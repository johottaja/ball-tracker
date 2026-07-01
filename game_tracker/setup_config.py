from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    DEFAULT_CAMERA_ANGLE_DEG,
    DEFAULT_HORIZONTAL_FOV_DEG,
    DEFAULT_MAIN_DISTANCE_M,
    DEFAULT_MAIN_HEIGHT_M,
    DEFAULT_SECONDARY_DISTANCE_M,
    DEFAULT_SECONDARY_HEIGHT_M,
    SETUP_JSON,
)


@dataclass
class CameraSetup:
    main_distance_m: float = DEFAULT_MAIN_DISTANCE_M
    secondary_distance_m: float = DEFAULT_SECONDARY_DISTANCE_M
    main_height_m: float = DEFAULT_MAIN_HEIGHT_M
    secondary_height_m: float = DEFAULT_SECONDARY_HEIGHT_M
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CameraSetup:
        return cls(
            main_distance_m=float(data.get("main_distance_m", DEFAULT_MAIN_DISTANCE_M)),
            secondary_distance_m=float(
                data.get("secondary_distance_m", DEFAULT_SECONDARY_DISTANCE_M)
            ),
            main_height_m=float(data.get("main_height_m", DEFAULT_MAIN_HEIGHT_M)),
            secondary_height_m=float(
                data.get("secondary_height_m", DEFAULT_SECONDARY_HEIGHT_M)
            ),
            camera_angle_deg=float(
                data.get("camera_angle_deg", DEFAULT_CAMERA_ANGLE_DEG)
            ),
            horizontal_fov_deg=float(
                data.get("horizontal_fov_deg", DEFAULT_HORIZONTAL_FOV_DEG)
            ),
        )


def default_setup() -> CameraSetup:
    return CameraSetup()


def load_setup_config(path: Path = SETUP_JSON) -> CameraSetup:
    if not path.is_file():
        return default_setup()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default_setup()
        return CameraSetup.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default_setup()


def save_setup_config(setup: CameraSetup, path: Path = SETUP_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(setup.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
