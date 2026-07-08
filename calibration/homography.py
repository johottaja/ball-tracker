from __future__ import annotations

import math

import cv2
import numpy as np

from .types import CameraCalibration, TableCalibration

# Typical webcam horizontal FOV used to score corner-order candidates.
_DEFAULT_HORIZONTAL_FOV_DEG = 70.0


def table_corner_world_coords(length_m: float, width_m: float) -> np.ndarray:
    """World (X, Y) for four table corners clicked clockwise when viewed from above (+Z).

    Origin is table center; +X along length, +Y along width.
    """
    half_l = length_m / 2.0
    half_w = width_m / 2.0
    return np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )


def compute_image_to_world_homography(
    image_corners: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Return 3×3 homography mapping image pixels to world table coordinates (X, Y, z=0)."""
    if len(image_corners) != 4:
        raise ValueError("Exactly four image corners are required")

    src = np.array(image_corners, dtype=np.float64)
    dst = table_corner_world_coords(length_m, width_m)
    homography, status = cv2.findHomography(src, dst)
    if homography is None or status is None:
        raise ValueError("Homography computation failed")
    return homography.astype(np.float64)


def order_quad_cyclic(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return four points in cyclic order around their centroid."""
    if len(points) != 4:
        raise ValueError("Exactly four points are required")

    centroid_x = sum(point[0] for point in points) / 4.0
    centroid_y = sum(point[1] for point in points) / 4.0
    return sorted(
        points,
        key=lambda point: math.atan2(point[1] - centroid_y, point[0] - centroid_x),
    )


def _focal_length_score(
    focal: float,
    *,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float = _DEFAULT_HORIZONTAL_FOV_DEG,
) -> float:
    """Higher is better. Rejects implausible focal lengths."""
    min_focal = 0.25 * max(image_width, image_height)
    max_focal = 5.0 * max(image_width, image_height)
    if focal < min_focal or focal > max_focal:
        return float("-inf")

    half_fov_rad = math.radians(horizontal_fov_deg / 2.0)
    ideal_focal = (image_width / 2.0) / math.tan(half_fov_rad)
    return -abs(math.log(focal / ideal_focal))


def resolve_corner_mapping(
    image_corners: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    """Match clicked corners to world table corners.

    Clicks may be in any order. Tries cyclic permutations of the quadrilateral
    and picks the assignment that yields a plausible focal length.
    """
    if len(image_corners) != 4:
        raise ValueError("Exactly four image corners are required")

    ordered = order_quad_cyclic(image_corners)
    principal_x = image_width / 2.0
    principal_y = image_height / 2.0

    best_corners: list[tuple[float, float]] | None = None
    best_score = float("-inf")
    last_error: ValueError | None = None

    for shift in range(4):
        rotated = ordered[shift:] + ordered[:shift]
        for reverse in (False, True):
            candidate = list(reversed(rotated)) if reverse else rotated
            try:
                homography = compute_image_to_world_homography(
                    candidate,
                    length_m=length_m,
                    width_m=width_m,
                )
                focal = estimate_focal_length(
                    homography,
                    principal_x=principal_x,
                    principal_y=principal_y,
                )
                score = _focal_length_score(
                    focal,
                    image_width=image_width,
                    image_height=image_height,
                )
                if score > best_score:
                    best_score = score
                    best_corners = candidate
            except ValueError as exc:
                last_error = exc

    if best_corners is None:
        message = (
            "Could not match table corners to a valid camera model. "
            "Click all four table corners so they form a clear quadrilateral."
        )
        if last_error is not None:
            raise ValueError(message) from last_error
        raise ValueError(message)

    return best_corners


def projection_matrix_from_corners(
    image_corners: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Compute a 3×4 projection matrix from four clicked table corners."""
    mapped_corners = resolve_corner_mapping(
        image_corners,
        length_m=length_m,
        width_m=width_m,
        image_width=image_width,
        image_height=image_height,
    )
    homography = compute_image_to_world_homography(
        mapped_corners,
        length_m=length_m,
        width_m=width_m,
    )
    intrinsic = intrinsic_matrix_from_homography(
        homography,
        width=image_width,
        height=image_height,
    )
    return projection_matrix_from_homography(homography, intrinsic)


def build_table_calibration(
    *,
    length_m: float,
    width_m: float,
    image_width: int,
    image_height: int,
    left_corners: list[tuple[float, float]],
    right_corners: list[tuple[float, float]],
) -> TableCalibration:
    return TableCalibration(
        table_length_m=length_m,
        table_width_m=width_m,
        image_width=image_width,
        image_height=image_height,
        cameras=[
            CameraCalibration(
                name="left",
                projection_matrix=projection_matrix_from_corners(
                    left_corners,
                    length_m=length_m,
                    width_m=width_m,
                    image_width=image_width,
                    image_height=image_height,
                ),
            ),
            CameraCalibration(
                name="right",
                projection_matrix=projection_matrix_from_corners(
                    right_corners,
                    length_m=length_m,
                    width_m=width_m,
                    image_width=image_width,
                    image_height=image_height,
                ),
            ),
        ],
    )


def _absolute_conic_form(
    h_a: np.ndarray,
    h_b: np.ndarray,
    principal_x: float,
    principal_y: float,
) -> float:
    """Bilinear form h_a^T B h_b, with B = K^-T K^-1 for square-pixel, zero-skew K.

    B factors into a term that depends only on the (known) principal point and a
    single unknown v = 1/f^2 (see estimate_focal_length for the derivation), so this
    helper returns the part of the form that multiplies v.
    """
    return (
        h_a[0] * h_b[0]
        + h_a[1] * h_b[1]
        - principal_x * (h_a[0] * h_b[2] + h_b[0] * h_a[2])
        - principal_y * (h_a[1] * h_b[2] + h_b[1] * h_a[2])
        + (principal_x**2 + principal_y**2) * h_a[2] * h_b[2]
    )


def estimate_focal_length(
    homography_image_to_world: np.ndarray,
    *,
    principal_x: float,
    principal_y: float,
) -> float:
    """Estimate a shared fx=fy from a single metric ground-plane homography.

    Assumes square pixels (no skew, fx = fy) and a principal point fixed at
    (principal_x, principal_y). Uses the standard orthogonality constraints on the
    homography's rotation columns (r1 . r2 = 0, |r1| = |r2|), solved as a 2x1 linear
    least-squares problem for v = 1/f^2.
    """
    homography_world_to_image = np.linalg.inv(homography_image_to_world)
    h1 = homography_world_to_image[:, 0]
    h2 = homography_world_to_image[:, 1]

    # r1 . r2 = 0  =>  a1 * v = b1
    a1 = _absolute_conic_form(h1, h2, principal_x, principal_y)
    b1 = -h1[2] * h2[2]

    # |r1| = |r2|  =>  a2 * v = b2
    a2 = _absolute_conic_form(h1, h1, principal_x, principal_y) - _absolute_conic_form(
        h2, h2, principal_x, principal_y
    )
    b2 = h2[2] ** 2 - h1[2] ** 2

    # Solve each orthogonality constraint independently (OpenCV autocalib style),
    # then prefer the estimate from the better-conditioned denominator.
    candidates: list[tuple[float, float]] = []
    if abs(a1) > 1e-12:
        v = b1 / a1
        if v > 0:
            candidates.append((v, abs(a1)))
    if abs(a2) > 1e-12:
        v = b2 / a2
        if v > 0:
            candidates.append((v, abs(a2)))

    if candidates:
        v = max(candidates, key=lambda item: item[1])[0]
        return math.sqrt(1.0 / v)

    denom = a1 * a1 + a2 * a2
    if denom < 1e-12:
        raise ValueError("Cannot estimate focal length from homography")

    v = (a1 * b1 + a2 * b2) / denom
    if v <= 0:
        raise ValueError("Invalid focal length from homography")
    return math.sqrt(1.0 / v)


def intrinsic_matrix_from_homography(
    homography_image_to_world: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    principal_x = width / 2.0
    principal_y = height / 2.0
    focal = estimate_focal_length(
        homography_image_to_world,
        principal_x=principal_x,
        principal_y=principal_y,
    )
    return np.array(
        [
            [focal, 0.0, principal_x],
            [0.0, focal, principal_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def projection_matrix_from_homography(
    homography_image_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Derive a 3×4 camera projection matrix from a ground-plane homography and intrinsics."""
    homography_world_to_image = np.linalg.inv(homography_image_to_world)
    m = np.linalg.inv(intrinsic) @ homography_world_to_image

    r1 = m[:, 0]
    r2 = m[:, 1]
    t = m[:, 2]

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    if r1_norm < 1e-9 or r2_norm < 1e-9:
        raise ValueError("Degenerate homography decomposition")

    scale = 0.5 * (r1_norm + r2_norm)
    r1 = r1 / scale
    r2 = r2 / scale
    t = t / scale

    r3 = np.cross(r1, r2)
    r3_norm = np.linalg.norm(r3)
    if r3_norm < 1e-9:
        raise ValueError("Degenerate rotation from homography")
    r3 = r3 / r3_norm

    rotation = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    extrinsic = np.hstack([rotation, t.reshape(3, 1)])
    return intrinsic @ extrinsic


def image_to_table_plane(
    homography_image_to_world: np.ndarray,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    """Map an image pixel to world (X, Y) on the table plane (z=0)."""
    point = homography_image_to_world @ np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-9:
        raise ValueError("Point maps to infinity on the table plane")
    return float(point[0] / point[2]), float(point[1] / point[2])

