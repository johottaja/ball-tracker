from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import RECORDINGS_DIR


def sanitize_training_set_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "unnamed"
    sanitized = re.sub(r"[^\w\- ]", "", name)
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "unnamed"


def training_set_dir(name: str) -> Path:
    return RECORDINGS_DIR / sanitize_training_set_name(name)


def next_clip_path(training_set_name: str) -> Path:
    directory = training_set_dir(training_set_name)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"clip_{stamp}.mp4"
