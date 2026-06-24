from pathlib import Path

from video_viewer.config import DISPLAY_MAX_SIZE, TARGET_RECORD_FPS

PACKAGE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PACKAGE_DIR / "recordings"
LEFT_VIDEO = RECORDINGS_DIR / "left.mp4"
RIGHT_VIDEO = RECORDINGS_DIR / "right.mp4"

STEREO_DISPLAY_MAX_SIZE = DISPLAY_MAX_SIZE
TARGET_FPS = TARGET_RECORD_FPS
