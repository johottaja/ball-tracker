from __future__ import annotations

import argparse
import sys
import tkinter as tk

from training_recorder.paths import training_set_dir

from throw_detection.labeller.app import ThrowLabellerApp
from throw_detection.labeller.clips import list_clips


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Label throw frames for clips in a training set.",
    )
    parser.add_argument(
        "set_name",
        help="Training set name (folder under recordings/)",
    )
    args = parser.parse_args(argv)

    set_dir = training_set_dir(args.set_name)
    if not set_dir.is_dir():
        print(
            f"Error: training set directory not found: {set_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not list_clips(args.set_name):
        print(
            f"Error: no clip_*.mp4 files in {set_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    root = tk.Tk()
    ThrowLabellerApp(root, args.set_name)
    root.mainloop()


if __name__ == "__main__":
    main()
