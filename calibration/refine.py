from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .homography import table_corner_world_coords
from .types import CameraCalibration, TableCalibration


def _intrinsic_from_projection(projection: np.ndarray) -> np.ndarray:
    intrinsic, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(projection)
    if abs(intrinsic[2, 2]) < 1e-9:
        raise ValueError("Cannot recover intrinsics from projection matrix")
    return intrinsic / intrinsic[2, 2]


def _refine_camera_projection(
    *,
    original_projection: np.ndarray,
    corner_pixels: list[tuple[float, float]],
    fingertip_pixels: list[tuple[float, float]],
    world_corners: np.ndarray,
) -> np.ndarray:
    if len(corner_pixels) != 4 or len(fingertip_pixels) != 4:
        raise ValueError("Four table-corner and four fingertip observations are required")

    intrinsic = _intrinsic_from_projection(original_projection)
    object_points = np.vstack((world_corners, world_corners)).astype(np.float64)
    image_points = np.array(corner_pixels + fingertip_pixels, dtype=np.float64)
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        image_points,
        intrinsic,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("Could not refine camera pose from fingertip observations")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return intrinsic @ np.hstack((rotation, translation))


def refine_calibration_from_fingertips(
    calibration: TableCalibration,
    *,
    left_corners: list[tuple[float, float]],
    right_corners: list[tuple[float, float]],
    left_fingertips: list[tuple[float, float]],
    right_fingertips: list[tuple[float, float]],
) -> TableCalibration:
    """Refine each camera's table-plane pose while preserving its calibrated intrinsics."""
    left = calibration.camera("left")
    right = calibration.camera("right")
    if left is None or right is None:
        raise ValueError("Calibration must include left and right cameras")

    world_corners_xy = table_corner_world_coords(
        calibration.table_length_m,
        calibration.table_width_m,
    )
    world_corners = np.column_stack(
        (world_corners_xy, np.zeros(len(world_corners_xy), dtype=np.float64))
    )
    refined_left = _refine_camera_projection(
        original_projection=left.projection_matrix,
        corner_pixels=left_corners,
        fingertip_pixels=left_fingertips,
        world_corners=world_corners,
    )
    refined_right = _refine_camera_projection(
        original_projection=right.projection_matrix,
        corner_pixels=right_corners,
        fingertip_pixels=right_fingertips,
        world_corners=world_corners,
    )
    return replace(
        calibration,
        cameras=[
            CameraCalibration(name="left", projection_matrix=refined_left),
            CameraCalibration(name="right", projection_matrix=refined_right),
        ],
        layout=None,
    )
