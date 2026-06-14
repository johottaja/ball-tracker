from __future__ import annotations

import numpy as np

from .config import (
    POSE_CONF_THRESHOLD,
    POSE_DEVICE,
    POSE_KEYPOINT_MIN_CONF,
    POSE_MODEL_PATH,
)
from .types import DominantHand, DominantHandDetection, HandSide, Joint

# COCO pose indices for each arm (shoulder → elbow → wrist).
_HAND_JOINTS: dict[HandSide, tuple[tuple[str, int], ...]] = {
    "left": (("shoulder", 5), ("elbow", 7), ("wrist", 9)),
    "right": (("shoulder", 6), ("elbow", 8), ("wrist", 10)),
}
_HIP_INDICES: dict[HandSide, int] = {"left": 11, "right": 12}


class PoseDetector:
    """Lazy-loaded YOLOv11 pose model."""

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(POSE_MODEL_PATH))
        return self._model

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return keypoint arrays shaped (17, 3) as x, y, confidence per person."""
        model = self._ensure_model()
        results = model(
            frame,
            conf=POSE_CONF_THRESHOLD,
            device=POSE_DEVICE,
            verbose=False,
        )
        if not results or results[0].keypoints is None:
            return []

        keypoints = results[0].keypoints.data.cpu().numpy()
        return [person for person in keypoints if person.shape[0] >= 17]


def _frame_center(frame: np.ndarray) -> tuple[float, float]:
    height, width = frame.shape[:2]
    return width / 2, height / 2


def _joint_from_keypoints(
    keypoints: np.ndarray,
    index: int,
    name: str,
) -> Joint | None:
    x, y, confidence = keypoints[index]
    if confidence < POSE_KEYPOINT_MIN_CONF:
        return None
    return Joint(name=name, x=float(x), y=float(y), confidence=float(confidence))


def _hand_joints(keypoints: np.ndarray, side: HandSide) -> tuple[Joint, ...] | None:
    joints: list[Joint] = []
    for name, index in _HAND_JOINTS[side]:
        joint = _joint_from_keypoints(keypoints, index, name)
        if joint is None:
            return None
        joints.append(joint)
    return tuple(joints)


def _wrist_distance_to_center(joints: tuple[Joint, ...], center: tuple[float, float]) -> float:
    wrist = joints[-1]
    return (wrist.x - center[0]) ** 2 + (wrist.y - center[1]) ** 2


def torso_segment(detection: DominantHandDetection) -> tuple[Joint, Joint] | None:
    """Dominant-side shoulder and hip joints in image coordinates."""
    side = detection.hand.side
    shoulder_index = _HAND_JOINTS[side][0][1]
    hip_index = _HIP_INDICES[side]
    shoulder = _joint_from_keypoints(
        detection.person_keypoints,
        shoulder_index,
        "shoulder",
    )
    hip = _joint_from_keypoints(detection.person_keypoints, hip_index, "hip")
    if shoulder is None or hip is None:
        return None
    return shoulder, hip


def torso_scale(person_keypoints: np.ndarray, side: HandSide) -> float | None:
    """Shoulder-to-hip length on the given side, used as a torso scale factor."""
    shoulder_index = _HAND_JOINTS[side][0][1]
    hip_index = _HIP_INDICES[side]
    shoulder = _joint_from_keypoints(person_keypoints, shoulder_index, "shoulder")
    hip = _joint_from_keypoints(person_keypoints, hip_index, "hip")
    if shoulder is None or hip is None:
        return None

    dx = hip.x - shoulder.x
    dy = hip.y - shoulder.y
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return None
    return length


def select_dominant_hand_detection(
    frame: np.ndarray,
    all_keypoints: list[np.ndarray],
) -> DominantHandDetection | None:
    """Pick the wrist closest to the frame center across all detected people."""
    center = _frame_center(frame)
    best_hand: DominantHand | None = None
    best_person_keypoints: np.ndarray | None = None
    best_distance = float("inf")

    for person_keypoints in all_keypoints:
        for side in ("left", "right"):
            joints = _hand_joints(person_keypoints, side)
            if joints is None:
                continue
            distance = _wrist_distance_to_center(joints, center)
            if distance < best_distance:
                best_distance = distance
                best_hand = DominantHand(side=side, joints=joints)
                best_person_keypoints = person_keypoints

    if best_hand is None or best_person_keypoints is None:
        return None
    return DominantHandDetection(hand=best_hand, person_keypoints=best_person_keypoints)


def select_dominant_hand(
    frame: np.ndarray,
    all_keypoints: list[np.ndarray],
) -> DominantHand | None:
    detection = select_dominant_hand_detection(frame, all_keypoints)
    if detection is None:
        return None
    return detection.hand


def detect_dominant_hand_detection(
    frame: np.ndarray,
    detector: PoseDetector | None = None,
) -> DominantHandDetection | None:
    pose_detector = detector or PoseDetector()
    keypoints = pose_detector.detect(frame)
    if not keypoints:
        return None
    return select_dominant_hand_detection(frame, keypoints)


def detect_dominant_hand(
    frame: np.ndarray,
    detector: PoseDetector | None = None,
) -> DominantHand | None:
    detection = detect_dominant_hand_detection(frame, detector=detector)
    if detection is None:
        return None
    return detection.hand
