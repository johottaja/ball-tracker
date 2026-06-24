from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageTk

from video_viewer.display import fit_size


def panel_size_for_frame(
    width: int,
    height: int,
    max_total_size: tuple[int, int],
) -> tuple[int, int]:
    max_w, max_h = max_total_size
    return fit_size(width, height, (max_w // 2, max_h))


def stereo_frame_to_photo(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    panel_size: tuple[int, int],
) -> ImageTk.PhotoImage:
    left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
    left_image = Image.fromarray(left_rgb).resize(panel_size, Image.Resampling.LANCZOS)
    right_image = Image.fromarray(right_rgb).resize(panel_size, Image.Resampling.LANCZOS)
    combined = Image.new("RGB", (panel_size[0] * 2, panel_size[1]))
    combined.paste(left_image, (0, 0))
    combined.paste(right_image, (panel_size[0], 0))
    return ImageTk.PhotoImage(combined)
