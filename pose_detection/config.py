from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL_PATH = REPO_ROOT / "yolo11n-pose.pt"

POSE_DEVICE = "mps"
POSE_CONF_THRESHOLD = 0.25
POSE_KEYPOINT_MIN_CONF = 0.3

# Longest image side passed to YOLO; keypoints are scaled back to full frame coords.
# Set to 0 to disable downscaling.
POSE_INFERENCE_MAX_SIZE = 640

# Shoulder → elbow → wrist for the dominant arm.
JOINT_NAMES = ("shoulder", "elbow", "wrist")
