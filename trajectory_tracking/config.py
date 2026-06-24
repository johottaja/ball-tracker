from __future__ import annotations

# Angular width of the ball-search sector, in degrees.
SECTOR_ANGLE_DEG: float = 150.0

# Center direction of the ball-search sector (screen coords: 0=right, 90=down).
# Default 135° = 45° counter-clockwise from left, tilting the wedge downward.
SECTOR_DIRECTION_DEG: float = 165.0

# Maximum distance (pixels) from the sector origin to accept a contour.
SECTOR_RADIUS_PX: int = 400

# Consecutive frames with no ball detection before the trajectory is finalised.
TRACKING_TIMEOUT_FRAMES: int = 3

# Trajectories with fewer detected ball positions are discarded (not shown).
MIN_TRAJECTORY_POINTS: int = 5

# Circularity range for contours accepted as ball candidates.
BALL_CIRCULARITY_MIN: float = 0.5
BALL_CIRCULARITY_MAX: float = 1.0

# Real-world torso length assumed for pixel-to-distance scaling.
ASSUMED_TORSO_CM: float = 50.0

# Frames averaged when smoothing shoulder-to-hip length for speed scaling.
TORSO_LENGTH_BUFFER_SIZE: int = 10
