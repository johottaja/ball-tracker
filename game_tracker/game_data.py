from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import GAME_JSON, LEFT_VIDEO, RIGHT_VIDEO

from calibration import TableCalibration

COORDINATE_SYSTEM = {
    "origin": "table_center",
    "x": "table_length",
    "y": "table_width",
    "z": "up_from_table",
    "units": "meters",
}


@dataclass
class Point2D:
    frame: int
    x: int
    y: int
    time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"frame": self.frame, "x": self.x, "y": self.y}
        if self.time_s is not None:
            d["time_s"] = self.time_s
        return d


@dataclass
class Point3D:
    frame: int | None
    x: float
    y: float
    z: float
    time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"x": self.x, "y": self.y, "z": self.z}
        if self.frame is not None:
            d["frame"] = self.frame
        if self.time_s is not None:
            d["time_s"] = self.time_s
        return d


@dataclass
class CurvePoint3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class ThrowRecord:
    id: int
    start_frame: int
    end_frame: int
    points_3d: list[Point3D]
    fitted_curve_3d: list[CurvePoint3D]
    speed_m_s: float | None
    tracks_2d: dict[str, list[Point2D]]
    thrower_side: str = "right"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "points_3d": [p.to_dict() for p in self.points_3d],
            "fitted_curve_3d": [p.to_dict() for p in self.fitted_curve_3d],
            "speed_m_s": self.speed_m_s,
            "thrower_side": self.thrower_side,
            "tracks_2d": {
                camera: [p.to_dict() for p in points]
                for camera, points in self.tracks_2d.items()
            },
        }


@dataclass
class GameSession:
    version: int = 1
    recorded_at: str = ""
    fps: float = 30.0
    frame_count: int = 0
    videos: dict[str, str] | None = None
    coordinate_system: dict[str, str] | None = None
    calibration: TableCalibration | None = None
    throws: list[ThrowRecord] | None = None

    def __post_init__(self) -> None:
        if not self.recorded_at:
            self.recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.videos is None:
            self.videos = {"left": LEFT_VIDEO.name, "right": RIGHT_VIDEO.name}
        if self.coordinate_system is None:
            self.coordinate_system = dict(COORDINATE_SYSTEM)
        if self.throws is None:
            self.throws = []

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "recorded_at": self.recorded_at,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "videos": self.videos,
            "coordinate_system": self.coordinate_system,
            "throws": [t.to_dict() for t in (self.throws or [])],
        }
        if self.calibration is not None:
            d["calibration"] = self.calibration.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameSession:
        throws: list[ThrowRecord] = []
        for item in data.get("throws", []):
            tracks_2d: dict[str, list[Point2D]] = {}
            for camera, points in item.get("tracks_2d", {}).items():
                tracks_2d[camera] = [
                    Point2D(
                        frame=p["frame"],
                        x=p["x"],
                        y=p["y"],
                        time_s=float(p["time_s"]) if "time_s" in p else None,
                    )
                    for p in points
                ]
            throws.append(
                ThrowRecord(
                    id=int(item["id"]),
                    start_frame=int(item["start_frame"]),
                    end_frame=int(item["end_frame"]),
                    points_3d=[
                        Point3D(
                            frame=p.get("frame"),
                            x=float(p["x"]),
                            y=float(p["y"]),
                            z=float(p["z"]),
                            time_s=float(p["time_s"]) if "time_s" in p else None,
                        )
                        for p in item.get("points_3d", [])
                    ],
                    fitted_curve_3d=[
                        CurvePoint3D(
                            x=float(p["x"]),
                            y=float(p["y"]),
                            z=float(p["z"]),
                        )
                        for p in item.get("fitted_curve_3d", [])
                    ],
                    speed_m_s=item.get("speed_m_s"),
                    tracks_2d=tracks_2d,
                    thrower_side=str(item.get("thrower_side", "right")),
                )
            )

        calibration_data = data.get("calibration")
        calibration = (
            TableCalibration.from_dict(calibration_data)
            if isinstance(calibration_data, dict)
            else None
        )

        return cls(
            version=int(data.get("version", 1)),
            recorded_at=str(data.get("recorded_at", "")),
            fps=float(data.get("fps", 30.0)),
            frame_count=int(data.get("frame_count", 0)),
            videos=data.get("videos"),
            coordinate_system=data.get("coordinate_system"),
            calibration=calibration,
            throws=throws,
        )


def new_game_session(
    *,
    fps: float,
    frame_count: int,
    calibration: TableCalibration | None = None,
) -> GameSession:
    return GameSession(
        recorded_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        fps=fps,
        frame_count=frame_count,
        calibration=calibration,
    )


def save_game(path: Path, session: GameSession) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_game(path: Path = GAME_JSON) -> GameSession | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return GameSession.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None
