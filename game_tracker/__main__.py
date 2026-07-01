import tkinter as tk

from .app import GameTrackerApp


def main() -> None:
    root = tk.Tk()
    GameTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
