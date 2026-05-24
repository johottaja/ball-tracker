from .app import VideoViewerApp

__all__ = ["VideoViewerApp"]


def main() -> None:
    from .__main__ import main as run

    run()
