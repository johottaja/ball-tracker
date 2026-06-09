import tkinter as tk

from .app import TrainingRecorderApp


def main() -> None:
    root = tk.Tk()
    TrainingRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
