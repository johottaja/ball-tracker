from __future__ import annotations

import math

import cv2

from pose_detection import (
    JOINT_NAMES,
    DominantHand,
    DominantHandDetection,
    PoseDetector,
    detect_dominant_hand,
    detect_dominant_hand_detection,
    normalize_hand_keypoints,
    torso_segment,
)
from throw_detection.inference import ThrowPrediction

from .config import POSE_BONE_THICKNESS, POSE_JOINT_RADIUS

_JOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "shoulder": (255, 0, 255),
    "elbow": (0, 255, 0),
    "wrist": (0, 255, 255),
}
_BONE_COLOR = (255, 255, 255)
_TORSO_SCALE_COLOR = (0, 165, 255)
_READOUT_BG_COLOR = (0, 0, 0)
_READOUT_TEXT_COLOR = (240, 240, 240)
_LABEL_ZERO_BG = (128, 128, 128)
_LABEL_ONE_BG = (0, 0, 220)
_LABEL_TEXT_COLOR = (255, 255, 255)
_LABEL_MARGIN = 16
_LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX
_LABEL_FONT_SCALE = 2.0
_LABEL_FONT_THICKNESS = 3
_LABEL_PAD = 12


def draw_dominant_hand(frame: np.ndarray, hand: DominantHand | None) -> np.ndarray:
    output = frame.copy()
    if hand is None:
        return output

    points: list[tuple[int, int]] = []
    for joint in hand.joints:
        point = (int(joint.x), int(joint.y))
        points.append(point)
        color = _JOINT_COLORS[joint.name]
        cv2.circle(output, point, POSE_JOINT_RADIUS, color, -1)
        cv2.circle(output, point, POSE_JOINT_RADIUS + 2, color, 2)

    for start, end in zip(points, points[1:]):
        cv2.line(output, start, end, _BONE_COLOR, POSE_BONE_THICKNESS)

    wrist = hand.joints[-1]
    label = f"{hand.side} hand"
    label_pos = (int(wrist.x) + 12, int(wrist.y) - 12)
    cv2.putText(
        output,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _JOINT_COLORS["wrist"],
        2,
        cv2.LINE_AA,
    )
    return output


def draw_torso_scale_line(
    frame: np.ndarray,
    detection: DominantHandDetection,
    *,
    scale: float | None = None,
) -> np.ndarray:
    segment = torso_segment(detection)
    if segment is None:
        return frame

    shoulder, hip = segment
    shoulder_point = (int(shoulder.x), int(shoulder.y))
    hip_point = (int(hip.x), int(hip.y))
    cv2.line(frame, shoulder_point, hip_point, _TORSO_SCALE_COLOR, POSE_BONE_THICKNESS)
    cv2.circle(frame, shoulder_point, POSE_JOINT_RADIUS, _TORSO_SCALE_COLOR, -1)
    cv2.circle(frame, hip_point, POSE_JOINT_RADIUS, _TORSO_SCALE_COLOR, -1)

    if scale is not None and not math.isnan(scale):
        midpoint = (
            (shoulder_point[0] + hip_point[0]) // 2,
            (shoulder_point[1] + hip_point[1]) // 2,
        )
        cv2.putText(
            frame,
            f"{scale:.0f}px",
            (midpoint[0] + 8, midpoint[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _TORSO_SCALE_COLOR,
            2,
            cv2.LINE_AA,
        )
    return frame


def _format_readout_value(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.2f}"


def draw_normalized_readout(
    frame: np.ndarray,
    normalized_keypoints: np.ndarray,
    scale: float,
) -> np.ndarray:
    lines = [
        *(
            f"{name}: ({_format_readout_value(normalized_keypoints[index, 0])}, "
            f"{_format_readout_value(normalized_keypoints[index, 1])}) "
            f"conf={_format_readout_value(normalized_keypoints[index, 2])}"
            for index, name in enumerate(JOINT_NAMES)
        ),
        f"torso scale: {_format_readout_value(scale)} px",
    ]

    margin = 10
    line_gap = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    height = frame.shape[0]
    y = height - margin

    for line in reversed(lines):
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        y -= text_height + baseline + line_gap
        top_left = (margin - 4, y - 2)
        bottom_right = (margin + text_width + 4, y + text_height + baseline + 2)
        cv2.rectangle(frame, top_left, bottom_right, _READOUT_BG_COLOR, -1)
        cv2.putText(
            frame,
            line,
            (margin, y + text_height),
            font,
            font_scale,
            _READOUT_TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    return frame


def apply_throw_detection(
    frame: np.ndarray,
    *,
    cache: object | None = None,
    frame_index: int | None = None,
    detector: PoseDetector | None = None,
) -> np.ndarray:
    if cache is not None and frame_index is not None:
        from .playback_cache import cached_pose_detection

        detection = cached_pose_detection(
            frame,
            detector=detector,
            cache=cache,
            frame_index=frame_index,
        )
        hand = detection.hand if detection is not None else None
    else:
        hand = detect_dominant_hand(frame, detector=detector)
    return draw_dominant_hand(frame, hand)


def draw_throw_label_badge(frame: np.ndarray, label: int) -> np.ndarray:
    output = frame.copy()
    text = str(int(label))
    bg_color = _LABEL_ONE_BG if label else _LABEL_ZERO_BG

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        _LABEL_FONT,
        _LABEL_FONT_SCALE,
        _LABEL_FONT_THICKNESS,
    )
    box_width = text_width + 2 * _LABEL_PAD
    box_height = text_height + baseline + 2 * _LABEL_PAD
    height, width = output.shape[:2]
    bottom_right = (width - _LABEL_MARGIN, height - _LABEL_MARGIN)
    top_left = (bottom_right[0] - box_width, bottom_right[1] - box_height)

    cv2.rectangle(output, top_left, bottom_right, bg_color, -1)
    text_origin = (
        top_left[0] + _LABEL_PAD,
        bottom_right[1] - _LABEL_PAD - baseline,
    )
    cv2.putText(
        output,
        text,
        text_origin,
        _LABEL_FONT,
        _LABEL_FONT_SCALE,
        _LABEL_TEXT_COLOR,
        _LABEL_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return output


def _label_badge_height() -> int:
    (text_width, text_height), baseline = cv2.getTextSize(
        "0",
        _LABEL_FONT,
        _LABEL_FONT_SCALE,
        _LABEL_FONT_THICKNESS,
    )
    return text_height + baseline + 2 * _LABEL_PAD


def draw_throw_prediction_readout(frame: np.ndarray, prediction: ThrowPrediction) -> np.ndarray:
    if prediction.has_pose:
        line = f"logit {prediction.logit:+.2f}  p={prediction.probability:.2f}"
    else:
        line = "no pose"

    margin = _LABEL_MARGIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    height = frame.shape[0]
    (text_width, text_height), baseline = cv2.getTextSize(
        line,
        font,
        font_scale,
        thickness,
    )
    y = height - margin - _label_badge_height() - 8
    x = frame.shape[1] - margin - text_width
    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 2),
        (x + text_width + 4, y + baseline + 2),
        _READOUT_BG_COLOR,
        -1,
    )
    cv2.putText(
        frame,
        line,
        (x, y),
        font,
        font_scale,
        _READOUT_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def apply_gru_throw_inference(
    frame: np.ndarray,
    prediction: ThrowPrediction,
) -> np.ndarray:
    detection = prediction.detection
    if detection is None:
        output = frame.copy()
    else:
        output = draw_dominant_hand(frame, detection.hand)
        normalized_keypoints, scale, _anchor = normalize_hand_keypoints(detection)
        output = draw_torso_scale_line(output, detection, scale=scale)
        output = draw_normalized_readout(output, normalized_keypoints, scale)

    output = draw_throw_prediction_readout(output, prediction)
    return draw_throw_label_badge(output, prediction.label)


def apply_normalized_throw_detection(
    frame: np.ndarray,
    *,
    detector: PoseDetector | None = None,
    cache: object | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    if cache is not None and frame_index is not None:
        from .playback_cache import cached_pose_detection

        detection = cached_pose_detection(
            frame,
            detector=detector,
            cache=cache,
            frame_index=frame_index,
        )
    else:
        detection = detect_dominant_hand_detection(frame, detector=detector)
    output = draw_dominant_hand(frame, detection.hand if detection else None)
    if detection is None:
        return output

    normalized_keypoints, scale, _anchor = normalize_hand_keypoints(detection)
    output = draw_torso_scale_line(output, detection, scale=scale)
    return draw_normalized_readout(output, normalized_keypoints, scale)
