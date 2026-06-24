import tkinter as tk

from .app import StereoViewerApp


def main() -> None:
    root = tk.Tk()
    StereoViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
