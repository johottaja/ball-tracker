# AGENTS.md

Guidance for AI agents working in this repository.

**Keep this file up to date.** When you add modules, change architecture, introduce dependencies, or shift project goals, update AGENTS.md in the same change so future agents have accurate context.

## Project overview

**balltracker** is a beer pong throw tracker. The long-term goal is to record a table from two cameras at different angles, track each ball’s trajectory through the throw, and display all throws on a 3D map.

**Current state:** Two Python desktop apps:

- **`video_viewer/`** — record webcam video and inspect it frame by frame. Includes computer-vision filters for debugging and two detection pipelines: **ball detection** (frame diff → threshold → circularity filter) and **throw detection** (YOLOv11 pose overlay via `pose_detection`).
- **`pose_detection/`** — reusable YOLO pose pipeline: per-frame dominant-hand selection and batch extraction of arm keypoints from frame sequences.
- **`training_recorder/`** — lightweight GUI for recording labeled training clips. Enter a training set name; each clip is saved under `recordings/<training_set>/` at the repo root (separate from `video_viewer/recordings/`).

Multi-camera capture, stereo triangulation, trajectory reconstruction, and 3D visualization are not implemented yet.

## Tech stack

- **Python 3.13+**, managed with [uv](https://github.com/astral-sh/uv) (`pyproject.toml`, `uv.lock`)
- **OpenCV** — camera capture, video I/O, image processing
- **Pillow** — frame conversion for tkinter display
- **tkinter** — GUI (stdlib)
- **Ultralytics** — YOLO pose model (`yolo11n-pose.pt`, gitignored; downloaded on first use)

## Running the app

```bash
uv sync
uv run python -m video_viewer
uv run python -m training_recorder
```

Alternative entry: `uv run python video_viewer/viewer.py`

`main.py` at the repo root is a placeholder; use `video_viewer` or `training_recorder` to run an app.

## Project structure

```
balltracker/
├── AGENTS.md                 # This file — agent context (keep updated)
├── README.md                 # User-facing setup and usage
├── pyproject.toml            # Dependencies and project metadata
├── uv.lock                   # Locked dependency versions
├── main.py                   # Placeholder entry point
├── recordings/               # Training clips by set name (gitignored)
├── yolo11n-pose.pt           # YOLO pose weights (gitignored, runtime download)
├── pose_detection/           # YOLO pose model, dominant-hand selection, batch extraction
│   ├── __init__.py
│   ├── config.py             # Model path, detection thresholds
│   ├── types.py              # Joint, DominantHand, DominantHandSequence
│   ├── detector.py           # PoseDetector, per-frame dominant-hand logic
│   ├── extract.py            # extract_dominant_hands / extract_normalized_dominant_hands
│   └── normalize.py          # Torso-relative shoulder→hip scaling
├── training_recorder/        # Training clip recorder GUI
│   ├── __init__.py
│   ├── __main__.py           # `python -m training_recorder` entry
│   ├── app.py                # tkinter UI: training set name, clip record/stop
│   ├── config.py             # Root recordings path, capture settings
│   └── paths.py              # Training set folder naming and clip paths
└── video_viewer/             # Viewer and CV debugging app
    ├── __init__.py
    ├── __main__.py           # `python -m video_viewer` entry
    ├── viewer.py             # Direct-run shim (adds parent to sys.path)
    ├── app.py                # tkinter UI: record/playback, filters, controls
    ├── camera.py             # Webcam open, FPS config, camera probing
    ├── config.py             # Paths and tuning constants
    ├── display.py            # Resize frames for UI display
    ├── recording.py          # VideoWriter helper
    ├── filters.py            # Filter registry and FrameFilter pipeline
    ├── ball_detection.py     # Contour/circularity logic and ball overlays
    ├── pose_overlay.py       # Dominant-hand skeleton overlay for the viewer filter
    └── recordings/           # Viewer default save dir (gitignored)
```

## Module responsibilities

| Module | Purpose |
|--------|---------|
| **training_recorder** | |
| `training_recorder/app.py` | `TrainingRecorderApp` — live preview, training set name, start/stop clips |
| `training_recorder/paths.py` | Sanitize set names; `recordings/<set>/clip_<timestamp>.mp4` |
| `training_recorder/config.py` | `RECORDINGS_DIR` at repo root; shared capture constants |
| **pose_detection** | |
| `detector.py` | Lazy YOLO load, dominant hand (shoulder→elbow→wrist) per frame |
| `extract.py` | `extract_dominant_hands` / `extract_normalized_dominant_hands` batch APIs |
| `normalize.py` | Shoulder-anchored, torso-scaled keypoint normalization |
| `types.py` | `Joint`, `DominantHand`, `DominantHandSequence` |
| `config.py` | `POSE_MODEL_PATH`, `POSE_CONF_THRESHOLD`, `POSE_KEYPOINT_MIN_CONF` |
| **video_viewer** | |
| `app.py` | `VideoViewerApp` — modes (record/playback), UI, frame stepping, filter wiring |
| `camera.py` | Open cameras (AVFoundation on macOS), probe indices, enforce min FPS |
| `config.py` | `RECORDINGS_DIR`, ball-detection thresholds, pose overlay drawing sizes |
| `filters.py` | `FilterId` enum, diff pipeline stages, `FrameFilter` state |
| `ball_detection.py` | Circular contour filtering, largest-ball selection, drawing |
| `pose_overlay.py` | Throw / normalized-throw filter overlays (imports `pose_detection`) |
| `recording.py` | Create MP4 writer at `recordings/recording.mp4` |
| `display.py` | Fit frames to max display size, convert to `PhotoImage` |

## Filter pipeline (ball detection)

Filters are display-only; recordings save raw camera frames.

1. **Frame difference** — `current − previous` (or mean of last N frames for window diff)
2. **Brightness amplification** — `DIFF_BRIGHTNESS_FACTOR`
3. **Threshold + morphological open** — `DIFF_THRESH_VALUE`, `MORPH_KERNEL_SIZE`
4. **Contour detection** — external contours on binary mask
5. **Circularity filter** — reject non-ball shapes via `BALL_CIRCULARITY_MIN/MAX`
6. **Largest contour** — treated as the ball; red rectangle on original frame

Intermediate diff filters exist for debugging each step. Ball detection needs a valid previous frame (first frame after seek shows nothing).

## Pose / throw detection

`pose_detection/` loads `yolo11n-pose.pt` on first use. For each frame it detects people, evaluates left/right arm chains (COCO keypoints 5–10), and picks the wrist closest to the frame center as the “dominant hand.”

`extract_dominant_hands(frames)` runs that selection over a frame list and returns a `DominantHandSequence`: `keypoints` shaped `(num_frames, 3, 3)` (shoulder/elbow/wrist × x, y, confidence) and `sides` (`-1` missing, `0` left, `1` right).

`extract_normalized_dominant_hands(frames)` returns a `NormalizedDominantHandSequence` with the same original `keypoints` plus `normalized_keypoints` (offset by dominant shoulder, scaled by same-side shoulder→hip length), `torso_scale`, and `anchor`.

The video viewer’s throw-detection filters draw joints and bones via `pose_overlay.py`. The normalized filter also draws the shoulder→hip scale line and a bottom-corner readout of torso-normalized coordinates.

Tune detection via `pose_detection/config.py`; overlay drawing via `video_viewer/config.py` (`POSE_JOINT_RADIUS`, `POSE_BONE_THICKNESS`).

## Configuration

**`video_viewer/config.py`**

- **Paths:** `RECORDINGS_DIR`, `DEFAULT_VIDEO`
- **Capture:** `TARGET_RECORD_FPS`, `MAX_CAMERA_PROBE`, `DISPLAY_MAX_SIZE`
- **Ball detection:** `DIFF_*`, `MORPH_KERNEL_SIZE`, `BALL_CIRCULARITY_*`, `DETECTION_RECT_THICKNESS`, `FRAME_WINDOW_SIZE`
- **Pose overlay:** `POSE_JOINT_RADIUS`, `POSE_BONE_THICKNESS`

**`pose_detection/config.py`**

- **Model:** `POSE_MODEL_PATH`, `POSE_CONF_THRESHOLD`, `POSE_KEYPOINT_MIN_CONF`

**`training_recorder/config.py`**

- **Paths:** `RECORDINGS_DIR` (repo root `recordings/`)
- **Capture:** `TARGET_RECORD_FPS`, `MAX_CAMERA_PROBE`, `DISPLAY_MAX_SIZE`

## Conventions

- Package code lives under `video_viewer/`, `training_recorder/`, and `pose_detection/`; keep detection logic separate from UI.
- Filters affect preview only unless explicitly designed to process recordings.
- Use `uv` for dependency changes (`uv add <package>`).
- Recorded videos and `.pt` model weights are gitignored.
- Prefer extending existing filter/detection modules over duplicating CV logic in `app.py`.

## Planned direction (not yet built)

- Dual-camera synchronized recording
- Ball position fusion across views → 3D trajectory
- Throw event detection (release point, arc, landing)
- 3D map UI showing historical throws

When implementing these, update this file and `README.md` to reflect new modules and workflows.
