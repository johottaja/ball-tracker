from __future__ import annotations

import cv2
import numpy as np

from pose_detection import PoseDetector
from video_viewer.pose_overlay import apply_normalized_throw_detection

_LABEL_ZERO_BG = (128, 128, 128)
_LABEL_ONE_BG = (0, 0, 220)
_LABEL_TEXT_COLOR = (255, 255, 255)
_MARGIN = 16
_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE = 2.0
_FONT_THICKNESS = 3
_PAD = 12


def draw_label_badge(frame: np.ndarray, label: int) -> np.ndarray:
    output = frame.copy()
    text = str(int(label))
    bg_color = _LABEL_ONE_BG if label else _LABEL_ZERO_BG

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        _FONT,
        _FONT_SCALE,
        _FONT_THICKNESS,
    )
    box_width = text_width + 2 * _PAD
    box_height = text_height + baseline + 2 * _PAD
    height, width = output.shape[:2]
    bottom_right = (width - _MARGIN, height - _MARGIN)
    top_left = (bottom_right[0] - box_width, bottom_right[1] - box_height)

    cv2.rectangle(output, top_left, bottom_right, bg_color, -1)
    text_origin = (
        top_left[0] + _PAD,
        bottom_right[1] - _PAD - baseline,
    )
    cv2.putText(
        output,
        text,
        text_origin,
        _FONT,
        _FONT_SCALE,
        _LABEL_TEXT_COLOR,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return output


def render_labeller_frame(
    frame: np.ndarray,
    label: int,
    *,
    detector: PoseDetector | None = None,
) -> np.ndarray:
    with_pose = apply_normalized_throw_detection(frame, detector=detector)
    return draw_label_badge(with_pose, label)
