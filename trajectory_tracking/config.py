from __future__ import annotations

# Angular width of the ball-search sector, in degrees.
SECTOR_ANGLE_DEG: float = 150.0

# Center direction of the ball-search sector (screen coords: 0=right, 90=down).
# Default 135° = 45° counter-clockwise from left, tilting the wedge downward.
SECTOR_DIRECTION_DEG: float = 165.0

# Maximum distance (pixels) from the sector origin to accept a contour.
SECTOR_RADIUS_PX: int = 400

# Consecutive frames with no ball detection before the trajectory is finalised.
TRACKING_TIMEOUT_FRAMES: int = 5

# Record this many arc points before upward motion / bounce counts as a miss.
BOUNCE_MISS_MIN_POINTS: int = 5

# Frames spent in SCANNING_BALL without finding a ball before giving up.
SCANNING_TIMEOUT_FRAMES: int = 10

# Frames in AWAITING_PARTNER (incl. discarded throws) before returning to idle.
AWAITING_PARTNER_TIMEOUT_FRAMES: int = 10

# Trajectories with fewer detected ball positions are discarded (not shown).
MIN_TRAJECTORY_POINTS: int = 5

# Circularity range for contours accepted as ball candidates (shared with ball detection).
BALL_CIRCULARITY_MIN: float = 0.4
BALL_CIRCULARITY_MAX: float = 1.0
BALL_CONTOUR_MIN_AREA: float = 100.0

# Real-world torso length assumed for pixel-to-distance scaling.
ASSUMED_TORSO_CM: float = 50.0

# Frames averaged when smoothing shoulder-to-hip length for speed scaling.
TORSO_LENGTH_BUFFER_SIZE: int = 10

# Palm estimate: elbow + PALM_EXTENSION * (wrist - elbow).
PALM_EXTENSION: float = 1.5

# Max frames to walk backward from first ball detection when finding release.
RELEASE_MAX_LOOKBACK_FRAMES: int = 45

# Accept release only if trajectory-palm distance is below this × forearm length.
RELEASE_HIT_RADIUS_FACTOR: float = 0.35
