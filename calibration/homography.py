from __future__ import annotations

import math

import cv2
import numpy as np

from .types import CameraCalibration, TableCalibration


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


def projection_matrix_from_corners(
    image_corners: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
) -> np.ndarray:
    """Compute a 3×4 projection matrix from four clicked table corners.

    Clicks must be in canonical world order (clockwise from above): (+L/2,+W/2),
    (+L/2,−W/2), (−L/2,−W/2), (−L/2,+W/2).

  When ``focal_length_px`` is set, that focal length is used instead of estimating
    it from the homography. Use this for digitally zoomed/cropped feeds where the
    table-plane fit is correct but the inferred FOV is too narrow.
    """
    if len(image_corners) != 4:
        raise ValueError("Exactly four image corners are required")

    homography = compute_image_to_world_homography(
        image_corners,
        length_m=length_m,
        width_m=width_m,
    )
    if focal_length_px is None:
        intrinsic = intrinsic_matrix_from_homography(
            homography,
            width=image_width,
            height=image_height,
        )
    else:
        intrinsic = intrinsic_matrix_from_focal_length(
            focal_length_px,
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
    match_right_focal_to_left: bool = False,
    right_horizontal_fov_deg: float | None = None,
) -> TableCalibration:
    left_homography = compute_image_to_world_homography(
        left_corners,
        length_m=length_m,
        width_m=width_m,
    )
    left_intrinsic = intrinsic_matrix_from_homography(
        left_homography,
        width=image_width,
        height=image_height,
    )
    left_focal_px = float(left_intrinsic[0, 0])

    if right_horizontal_fov_deg is not None:
        right_focal_px = focal_length_from_horizontal_fov(
            right_horizontal_fov_deg,
            image_width=image_width,
        )
    elif match_right_focal_to_left:
        right_focal_px = left_focal_px
    else:
        right_focal_px = None

    return TableCalibration(
        table_length_m=length_m,
        table_width_m=width_m,
        image_width=image_width,
        image_height=image_height,
        cameras=[
            CameraCalibration(
                name="left",
                projection_matrix=projection_matrix_from_homography(
                    left_homography,
                    left_intrinsic,
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
                    focal_length_px=right_focal_px,
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


def intrinsic_matrix_from_focal_length(
    focal: float,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    principal_x = width / 2.0
    principal_y = height / 2.0
    return np.array(
        [
            [focal, 0.0, principal_x],
            [0.0, focal, principal_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def focal_length_from_horizontal_fov(
    horizontal_fov_deg: float,
    *,
    image_width: int,
) -> float:
    if horizontal_fov_deg <= 0.0 or horizontal_fov_deg >= 179.0:
        raise ValueError("Horizontal FOV must be between 0° and 179°")
    half_fov_rad = math.radians(horizontal_fov_deg / 2.0)
    return (image_width / 2.0) / math.tan(half_fov_rad)


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
    """Derive a 3×4 projection matrix from a ground-plane homography and intrinsics.

    A plane homography is unchanged when all of its decomposed columns are
    negated.  Those two signs put the recovered camera center on opposite sides
    of the table, so retain the branch whose center is above the Z=0 table
    plane.
    """
    homography_world_to_image = np.linalg.inv(homography_image_to_world)
    m = np.linalg.inv(intrinsic) @ homography_world_to_image

    homography_r1 = m[:, 0]
    homography_r2 = m[:, 1]
    homography_t = m[:, 2]

    r1_norm = np.linalg.norm(homography_r1)
    r2_norm = np.linalg.norm(homography_r2)
    if r1_norm < 1e-9 or r2_norm < 1e-9:
        raise ValueError("Degenerate homography decomposition")

    scale = 0.5 * (r1_norm + r2_norm)
    base_r1 = homography_r1 / scale
    base_r2 = homography_r2 / scale
    base_t = homography_t / scale

    candidates: list[tuple[float, np.ndarray]] = []
    for sign in (1.0, -1.0):
        r1 = sign * base_r1
        r2 = sign * base_r2
        t = sign * base_t

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

        camera_center = -rotation.T @ t
        extrinsic = np.hstack([rotation, t.reshape(3, 1)])
        candidates.append((float(camera_center[2]), intrinsic @ extrinsic))

    above_table = [candidate for candidate in candidates if candidate[0] > 1e-9]
    if not above_table:
        raise ValueError("Homography decomposition places the camera on or below the table")
    return max(above_table, key=lambda candidate: candidate[0])[1]


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


def _world_table_to_image_px(
    world_to_image: np.ndarray,
    world_x: float,
    world_y: float,
) -> tuple[float, float] | None:
    point = world_to_image @ np.array([world_x, world_y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-9:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def _draw_world_table_line(
    image: np.ndarray,
    world_to_image: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    start = _world_table_to_image_px(world_to_image, start_xy[0], start_xy[1])
    end = _world_table_to_image_px(world_to_image, end_xy[0], end_xy[1])
    if start is None or end is None:
        return

    height, width = image.shape[:2]
    pt1 = (int(round(start[0])), int(round(start[1])))
    pt2 = (int(round(end[0])), int(round(end[1])))
    ok, clipped_pt1, clipped_pt2 = cv2.clipLine((0, 0, width - 1, height - 1), pt1, pt2)
    if not ok:
        return
    cv2.line(image, clipped_pt1, clipped_pt2, color_bgr, thickness, lineType=cv2.LINE_AA)


def _grid_world_bounds_for_image(
    image_to_world: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    length_m: float,
    width_m: float,
    cell_size_m: float,
) -> tuple[float, float, float, float]:
    """World XY span that covers the image frame on the table plane, snapped to grid cells."""
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    world_points: list[tuple[float, float]] = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ]

    for pixel_x, pixel_y in (
        (0.0, 0.0),
        (float(image_width), 0.0),
        (float(image_width), float(image_height)),
        (0.0, float(image_height)),
    ):
        try:
            world_points.append(image_to_table_plane(image_to_world, pixel_x, pixel_y))
        except ValueError:
            continue

    min_x = min(point[0] for point in world_points)
    max_x = max(point[0] for point in world_points)
    min_y = min(point[1] for point in world_points)
    max_y = max(point[1] for point in world_points)

    min_x = math.floor(min_x / cell_size_m) * cell_size_m - cell_size_m
    max_x = math.ceil(max_x / cell_size_m) * cell_size_m + cell_size_m
    min_y = math.floor(min_y / cell_size_m) * cell_size_m - cell_size_m
    max_y = math.ceil(max_y / cell_size_m) * cell_size_m + cell_size_m
    return min_x, min_y, max_x, max_y


def draw_table_xy_grid_on_image(
    image: np.ndarray,
    image_corners: list[tuple[float, float]],
    *,
    length_m: float,
    width_m: float,
    cell_size_m: float = 0.15,
    color_bgr: tuple[int, int, int] = (239, 207, 158),
    thickness: int = 1,
) -> np.ndarray:
    """Overlay metric XY grid lines on the table plane, projected into the image."""
    if len(image_corners) != 4 or cell_size_m <= 0:
        return image

    try:
        image_to_world = compute_image_to_world_homography(
            image_corners,
            length_m=length_m,
            width_m=width_m,
        )
    except ValueError:
        return image

    world_to_image = np.linalg.inv(image_to_world)
    height, width = image.shape[:2]
    min_x, min_y, max_x, max_y = _grid_world_bounds_for_image(
        image_to_world,
        image_width=width,
        image_height=height,
        length_m=length_m,
        width_m=width_m,
        cell_size_m=cell_size_m,
    )
    out = image.copy()

    y = min_y
    while y <= max_y + 1e-9:
        _draw_world_table_line(
            out,
            world_to_image,
            (min_x, y),
            (max_x, y),
            color_bgr=color_bgr,
            thickness=thickness,
        )
        y += cell_size_m

    x = min_x
    while x <= max_x + 1e-9:
        _draw_world_table_line(
            out,
            world_to_image,
            (x, min_y),
            (x, max_y),
            color_bgr=color_bgr,
            thickness=thickness,
        )
        x += cell_size_m

    return out

