from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import GAMES_DIR


def default_game_json_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"game-{stamp}.json"


def sanitize_game_json_filename(name: str) -> str:
    name = name.strip()
    if not name:
        return default_game_json_name()
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    stem = name[:-5]
    sanitized = re.sub(r"[^\w\-]", "", stem.replace(" ", "-"))
    sanitized = sanitized or "game"
    return f"{sanitized}.json"


def game_json_path(name: str) -> Path:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    return GAMES_DIR / sanitize_game_json_filename(name)


def latest_game_json() -> Path | None:
    if not GAMES_DIR.is_dir():
        return None
    candidates = sorted(
        GAMES_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
