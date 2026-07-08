from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraCalibration:
    """Per-camera 3×4 projection matrix (world XYZ → image pixels, homogeneous)."""

    name: str
    projection_matrix: np.ndarray

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "projection_matrix": self.projection_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraCalibration:
        matrix = np.array(data["projection_matrix"], dtype=np.float64)
        if matrix.shape != (3, 4):
            raise ValueError(
                f"Camera {data.get('name', '?')}: projection_matrix must be 3×4"
            )
        return cls(
            name=str(data["name"]),
            projection_matrix=matrix,
        )


@dataclass
class TableCalibration:
    table_length_m: float
    table_width_m: float
    image_width: int
    image_height: int
    cameras: list[CameraCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "table_length_m": self.table_length_m,
            "table_width_m": self.table_width_m,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "cameras": [camera.to_dict() for camera in self.cameras],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TableCalibration:
        return cls(
            table_length_m=float(data["table_length_m"]),
            table_width_m=float(data["table_width_m"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            cameras=[CameraCalibration.from_dict(item) for item in data.get("cameras", [])],
        )

    def camera(self, name: str) -> CameraCalibration | None:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None
