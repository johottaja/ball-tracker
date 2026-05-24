"""Entry point for: python video_viewer/viewer.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly without installing the package.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from video_viewer.__main__ import main

    main()
