from __future__ import annotations

import cv2
from PIL import Image, ImageTk

from .config import DISPLAY_MAX_SIZE


def fit_size(
    width: int,
    height: int,
    max_size: tuple[int, int] = DISPLAY_MAX_SIZE,
) -> tuple[int, int]:
    max_w, max_h = max_size
    scale = min(max_w / width, max_h / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale))


def frame_to_photo(frame, display_size: tuple[int, int]) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image = image.resize(display_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)
