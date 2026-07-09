from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
class CameraLayoutStats:
    name: str
    center: tuple[float, float, float]
    xy_distance_m: float
    z_m: float
    yaw_deg: float
    pitch_deg: float
    horizontal_fov_deg: float
    fov_left_xy: tuple[float, float] | None
    fov_right_xy: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "center": list(self.center),
            "xy_distance_m": self.xy_distance_m,
            "z_m": self.z_m,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "horizontal_fov_deg": self.horizontal_fov_deg,
        }
        if self.fov_left_xy is not None:
            d["fov_left_xy"] = list(self.fov_left_xy)
        if self.fov_right_xy is not None:
            d["fov_right_xy"] = list(self.fov_right_xy)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CameraLayoutStats:
        center = data["center"]
        fov_left = data.get("fov_left_xy")
        fov_right = data.get("fov_right_xy")
        return cls(
            name=str(data["name"]),
            center=(float(center[0]), float(center[1]), float(center[2])),
            xy_distance_m=float(data["xy_distance_m"]),
            z_m=float(data["z_m"]),
            yaw_deg=float(data["yaw_deg"]),
            pitch_deg=float(data["pitch_deg"]),
            horizontal_fov_deg=float(data["horizontal_fov_deg"]),
            fov_left_xy=(float(fov_left[0]), float(fov_left[1])) if fov_left else None,
            fov_right_xy=(float(fov_right[0]), float(fov_right[1])) if fov_right else None,
        )


@dataclass
class StereoLayoutStats:
    baseline_xy_m: float
    baseline_3d_m: float
    delta_z_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "baseline_xy_m": self.baseline_xy_m,
            "baseline_3d_m": self.baseline_3d_m,
            "delta_z_m": self.delta_z_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StereoLayoutStats:
        return cls(
            baseline_xy_m=float(data["baseline_xy_m"]),
            baseline_3d_m=float(data["baseline_3d_m"]),
            delta_z_m=float(data["delta_z_m"]),
        )


@dataclass
class CalibrationLayout:
    cameras: list[CameraLayoutStats]
    stereo: StereoLayoutStats | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cameras": [camera.to_dict() for camera in self.cameras],
        }
        if self.stereo is not None:
            d["stereo"] = self.stereo.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationLayout:
        stereo_data = data.get("stereo")
        return cls(
            cameras=[
                CameraLayoutStats.from_dict(item) for item in data.get("cameras", [])
            ],
            stereo=StereoLayoutStats.from_dict(stereo_data) if stereo_data else None,
        )


@dataclass
class TableCalibration:
    table_length_m: float
    table_width_m: float
    image_width: int
    image_height: int
    cameras: list[CameraCalibration] = field(default_factory=list)
    layout: CalibrationLayout | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "table_length_m": self.table_length_m,
            "table_width_m": self.table_width_m,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "cameras": [camera.to_dict() for camera in self.cameras],
        }
        if self.layout is not None:
            d["layout"] = self.layout.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TableCalibration:
        layout_data = data.get("layout")
        return cls(
            table_length_m=float(data["table_length_m"]),
            table_width_m=float(data["table_width_m"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            cameras=[CameraCalibration.from_dict(item) for item in data.get("cameras", [])],
            layout=CalibrationLayout.from_dict(layout_data) if layout_data else None,
        )

    def camera(self, name: str) -> CameraCalibration | None:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None
