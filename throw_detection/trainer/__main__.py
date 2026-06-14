from __future__ import annotations

import tkinter as tk

from throw_detection.trainer.app import ThrowTrainerApp


def main() -> None:
    root = tk.Tk()
    ThrowTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
