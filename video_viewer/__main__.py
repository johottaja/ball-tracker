import tkinter as tk

from .app import VideoViewerApp


def main() -> None:
    root = tk.Tk()
    VideoViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
