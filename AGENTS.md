# AGENTS.md

Guidance for AI agents working in this repository.

**Keep this file up to date.** When you add modules, change architecture, introduce dependencies, or shift project goals, update AGENTS.md in the same change so future agents have accurate context.

## Project overview

**balltracker** is a beer pong throw tracker. The long-term goal is to record a table from two cameras at different angles, track each ball’s trajectory through the throw, and display all throws on a 3D map.

**Current state:** Five Python desktop apps plus shared detection libraries:

- **`video_viewer/`** — record webcam video and inspect it frame by frame. Includes configurable **ball detection** (MOG2 + morphological closing, or frame diff → threshold) with contour/circularity filtering, plus **throw detection** (YOLOv11 pose overlay via `pose_detection`).
- **`stereo_viewer/`** — dual-camera version of the video viewer: side-by-side live preview and playback, same filter set applied independently per camera (plus **Stereo tracking** and **Frame sync**, stereo-only). Records synchronized `left.mp4` and `right.mp4` under `stereo_viewer/recordings/`.
- **`pose_detection/`** — reusable YOLO pose pipeline: per-frame dominant-hand selection and batch extraction of arm keypoints from frame sequences.
- **`training_recorder/`** — lightweight GUI for recording labeled training clips. Enter a training set name; each clip is saved under `recordings/<training_set>/` at the repo root (separate from `video_viewer/recordings/`).
- **`throw_detection/`** — throw-event labeling GUI, GRU training-data export, GRU training GUI, and streaming GRU inference. Labels per-frame throw/not-throw on clips from `recordings/<set>/`; saves NumPy `.npz` datasets under `throw_detection/training_sets/`; trained models under `throw_detection/models/`.
- **`trajectory_tracking/`** — stateful ball trajectory tracker that combines throw inference with configurable ball motion masks. Three phases: detecting throw → scanning for ball in a circular sector from the wrist → tracking ball frame-by-frame. Fits a parabola to the collected positions and exposes drawing helpers for the video viewer filter.
- **`framesync/`** — stereo camera frame-offset measurement from deliberate straight-down ball drops and table bounces. Per-camera macro phase machine plus subframe bounce-time estimation; reused by the stereo viewer **Frame sync** filter.

Dual-camera synchronized recording is available via `stereo_viewer`. Stereo triangulation, trajectory reconstruction, and 3D visualization are not implemented yet.

## Tech stack

- **Python 3.13+**, managed with [uv](https://github.com/astral-sh/uv) (`pyproject.toml`, `uv.lock`)
- **OpenCV** — camera capture, video I/O, image processing
- **Pillow** — frame conversion for tkinter display
- **tkinter** — GUI (stdlib)
- **Ultralytics** — YOLO pose model (`yolo11n-pose.pt`, gitignored; downloaded on first use)
- **PyTorch** — GRU throw classifier training (`throw_detection/trainer`)

## Running the app

```bash
uv sync
uv run python -m video_viewer
uv run python -m stereo_viewer
uv run python -m training_recorder
uv run python -m throw_detection.labeller <set_name>
uv run python -m throw_detection.trainer
```

Alternative entry: `uv run python video_viewer/viewer.py`

`main.py` at the repo root is a placeholder; use `video_viewer`, `stereo_viewer`, `training_recorder`, `throw_detection.labeller`, or `throw_detection.trainer` to run an app.

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
├── throw_detection/          # Throw labeling + GRU training
│   ├── __init__.py
│   ├── config.py             # BUFFER_SIZE, TRAINING_SETS_DIR, MODELS_DIR
│   ├── features.py           # elbow/wrist features + rolling windows
│   ├── dataset.py            # LabelingSession, .npz save/load
│   ├── model.py              # ThrowGRU, save/load checkpoints
│   ├── train.py              # clip-level train/val split, training loop
│   ├── inference.py          # streaming GRU throw classifier (single-frame API)
│   ├── labeller/
│   │   ├── __main__.py       # `python -m throw_detection.labeller` entry
│   │   ├── app.py            # tkinter labeling UI
│   │   ├── clips.py          # list/load clips, pose extraction
│   │   └── overlay.py        # normalized pose overlay + label badge
│   ├── trainer/
│   │   ├── __main__.py       # `python -m throw_detection.trainer` entry
│   │   └── app.py            # tkinter training UI (hyperparams, progress, save)
│   ├── training_sets/        # Saved .npz datasets (gitignored)
│   └── models/               # Saved GRU checkpoints (gitignored)
├── trajectory_tracking/      # Ball trajectory tracker
│   ├── __init__.py
│   ├── config.py             # sector, tracking, torso/speed constants
│   ├── speed.py              # TorsoLengthBuffer, curve-length speed estimate
│   ├── tracker.py            # Phase enum, TrajectoryResult, TrajectoryTracker
│   └── drawing.py            # draw_trajectory_overlay (sector, points, parabola, speed)
├── framesync/                # Stereo frame-offset from ball drop/bounce
│   ├── __init__.py
│   ├── config.py             # drop/bounce thresholds, session timeouts
│   ├── types.py              # Phase, BallSample, FrameSyncResult
│   ├── tracker.py            # CameraSyncTracker (per-camera state machine)
│   ├── engine.py             # FrameSyncEngine (stereo session + offset math)
│   ├── subframe.py           # estimate_bounce_subframe_index
│   └── drawing.py            # draw_framesync_overlay (sync label, phase)
├── stereo_viewer/            # Dual-camera viewer (side-by-side)
│   ├── __init__.py
│   ├── __main__.py           # `python -m stereo_viewer` entry
│   ├── app.py                # StereoViewerApp — two cameras, shared controls/filters
│   ├── config.py             # LEFT_VIDEO, RIGHT_VIDEO, stereo display size
│   ├── display.py            # Side-by-side frame compositing for tkinter
│   ├── stereo_tracking.py    # Stereo tracking filter: main GRU + secondary ball trajectory
│   ├── frame_sync.py         # Frame sync filter: FrameSyncProcessor wrapper
│   └── recordings/           # Default left.mp4 / right.mp4 (gitignored)
└── video_viewer/             # Viewer and CV debugging app
    ├── __init__.py
    ├── __main__.py           # `python -m video_viewer` entry
    ├── viewer.py             # Direct-run shim (adds parent to sys.path)
    ├── app.py                # VideoViewerApp — record/playback, filters, controls
    ├── camera.py             # Webcam open, FPS config, camera probing
    ├── config.py             # Paths and tuning constants
    ├── display.py            # Resize frames for UI display
    ├── filter_controls.py    # Shared filter + ball-detection method comboboxes, Filters menu
    ├── playback.py           # Shared playback helpers (seek, motion-mask/GRU context, render)
    ├── recording.py          # VideoWriter helper
    ├── filters.py            # Filter registry and FrameFilter pipeline
    ├── ball_motion.py        # BallDetectionMethod, MotionMaskBuilder (MOG2 / frame diff)
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
| `config.py` | `POSE_MODEL_PATH`, `POSE_DEVICE`, `POSE_CONF_THRESHOLD`, `POSE_KEYPOINT_MIN_CONF` |
| **stereo_viewer** | |
| `stereo_viewer/app.py` | `StereoViewerApp` — two camera streams, side-by-side preview/playback, independent `FrameFilter` per camera (or coordinated **Stereo tracking**) |
| `stereo_viewer/config.py` | `RECORDINGS_DIR`, `LEFT_VIDEO`, `RIGHT_VIDEO`, `STEREO_DISPLAY_MAX_SIZE` |
| `stereo_viewer/display.py` | `panel_size_for_frame`, `stereo_frame_to_photo` (horizontal composite) |
| `stereo_viewer/stereo_tracking.py` | `StereoTrackingProcessor` — main GRU + ball track on both cameras; secondary ball-only track |
| `stereo_viewer/frame_sync.py` | `FrameSyncProcessor` — ball drop/bounce sync on both cameras via `FrameSyncEngine` |
| **video_viewer** | |
| `app.py` | `VideoViewerApp` — modes (record/playback), UI, frame stepping, filter wiring |
| `filter_controls.py` | `FilterControls` — filter combobox, ball-detection method combobox, Filters menu (both viewers) |
| `playback.py` | Seek helpers, motion-mask/GRU warmup context, `frame_to_display_photo` |
| `camera.py` | Open cameras (AVFoundation on macOS), probe indices, enforce min FPS; `CameraReader` captures on a background thread |
| `config.py` | `RECORDINGS_DIR`, ball-motion thresholds (MOG2, frame diff), pose overlay drawing sizes |
| `filters.py` | `FilterId` enum, `FrameFilter` state |
| `ball_motion.py` | `BallDetectionMethod`, `MotionMaskBuilder` — MOG2 + closing or frame diff masks |
| `ball_detection.py` | Circular contour filtering, largest-ball selection, `contour_bottom_center`, drawing |
| `pose_overlay.py` | Throw / normalized-throw / GRU-inference filter overlays (imports `pose_detection`, `throw_detection.inference`) |
| `recording.py` | Create MP4 writer at `recordings/recording.mp4` |
| `display.py` | Fit frames to max display size, convert to `PhotoImage` |
| **throw_detection** | |
| `config.py` | `BUFFER_SIZE` (GRU rolling window), `TRAINING_SETS_DIR`, `MODELS_DIR` |
| `features.py` | `frame_features_from_sequence`, `rolling_windows` |
| `dataset.py` | `LabelingSession`, `save_dataset` / `load_dataset`, resume labels |
| `model.py` | `ThrowGRU`, `save_throw_model` / `load_throw_model` |
| `train.py` | `train_throw_model`, clip-level validation split, metrics |
| `labeller/app.py` | `ThrowLabellerApp` — playback, per-frame 0/1 labels, clip nav, save |
| `labeller/clips.py` | `list_clips`, `read_frame_at`, `extract_pose_from_video` (streamed, on save) |
| `labeller/overlay.py` | Normalized pose overlay + bottom-right label badge |
| `trainer/app.py` | `ThrowTrainerApp` — pick `.npz` set, tune hyperparams, train, save `.pt` |
| `inference.py` | `ThrowInference` — load `.pt`, rolling feature window, per-frame `ThrowPrediction` |
| **trajectory_tracking** | |
| `tracker.py` | `TrajectoryTracker` — three-phase state machine (detecting throw → scanning ball → tracking ball); fits parabola on trajectory exit; counts tracking frames per throw |
| `speed.py` | `TorsoLengthBuffer` (10-frame rolling mean of shoulder→hip px); `estimate_throw_speed_m_s` from fitted curve length × torso scale ÷ tracking duration |
| `drawing.py` | `draw_trajectory_overlay` — sector wedge, ball markers, active/completed points, fitted parabola, phase label, completed throw speed (top-right) |
| `config.py` | `SECTOR_ANGLE_DEG`, `SECTOR_DIRECTION_DEG`, `SECTOR_RADIUS_PX`, `TRACKING_TIMEOUT_FRAMES`, `BALL_CIRCULARITY_MIN/MAX`, `ASSUMED_TORSO_CM`, `TORSO_LENGTH_BUFFER_SIZE` |
| **framesync** | |
| `tracker.py` | `CameraSyncTracker` — per-camera phases: watching → syncing → capturing → done |
| `engine.py` | `FrameSyncEngine` — pairs sync sessions across cameras, computes offset when both finish |
| `subframe.py` | `estimate_bounce_subframe_index` — velocity zero-crossing / quadratic peak within bounce frame pair |
| `drawing.py` | `draw_framesync_overlay` — large top-center ±offset label, phase readout, ball bottom marker |
| `config.py` | `DROP_STREAK_FRAMES`, `MAX_HORIZONTAL_DELTA_PX`, `POST_BOUNCE_CAPTURE_FRAMES`, `SLOWDOWN_RATIO`, `MIN_DOWNWARD_VY`, `SYNC_TIMEOUT_FRAMES`, `SYNC_PAIRING_WINDOW_FRAMES`, `SYNC_COOLDOWN_SECONDS` |

## Ball detection

Both viewers expose a **Ball detection** dropdown (independent of the display filter). The selected method builds a binary motion mask used by **Contours**, **Ball detection**, **Trajectory tracking**, **Stereo tracking**, and **Frame sync**.

**MOG2 + morphological closing (default):** `cv2.createBackgroundSubtractorMOG2` foreground mask, then morphological close (dilation + erosion). On playback seeks, prior frames up to `MOG2_HISTORY` are fed through the subtractor before the current frame.

**Frame diff:** `current − previous`, brightness amplification (`DIFF_BRIGHTNESS_FACTOR`), threshold (`DIFF_THRESH_VALUE`), morphological open (`FRAME_DIFF_MORPH_KERNEL_SIZE`). Needs the previous frame (or sequential streaming state).

Shared contour step (both methods):

1. **Contour detection** — external contours on the binary mask
2. **Circularity filter** — reject non-ball shapes via `BALL_CIRCULARITY_MIN/MAX`
3. **Minimum area** — reject contours smaller than `BALL_CONTOUR_MIN_AREA` px²
4. **Largest contour** — treated as the ball

**Contours** filter draws all circular contours on the mask. **Ball detection** draws a red bounding rectangle on the original frame.

Filters are display-only; recordings save raw camera frames.

## Pose / throw detection

`pose_detection/` loads `yolo11n-pose.pt` on first use. For each frame it detects people, evaluates left/right arm chains (COCO keypoints 5–10), and picks the wrist closest to the frame center as the “dominant hand.”

`extract_dominant_hands(frames)` runs that selection over a frame list and returns a `DominantHandSequence`: `keypoints` shaped `(num_frames, 3, 3)` (shoulder/elbow/wrist × x, y, confidence) and `sides` (`-1` missing, `0` left, `1` right).

`extract_normalized_dominant_hands(frames)` returns a `NormalizedDominantHandSequence` with the same original `keypoints` plus `normalized_keypoints` (offset by dominant shoulder, scaled by same-side shoulder→hip length), `torso_scale`, and `anchor`.

The video viewer’s throw-detection filters draw joints and bones via `pose_overlay.py`. The normalized filter also draws the shoulder→hip scale line and a bottom-corner readout of torso-normalized coordinates.

Tune detection via `pose_detection/config.py`; overlay drawing via `video_viewer/config.py` (`POSE_JOINT_RADIUS`, `POSE_BONE_THICKNESS`).

## Throw labeling (`throw_detection.labeller`)

Loads all `clip_*.mp4` files from `recordings/<set_name>/` via OpenCV seek (no full-clip RAM buffer). Clips open immediately; pose batch extraction runs on **Save** only. Each frame starts labeled `0` (not throwing). **Space** toggles: `0→1` advances one frame; `1→0` stays on the same frame. Arrow keys step/play/pause; **Save** writes `throw_detection/training_sets/<set_name>.npz`.

Display uses the same normalized pose overlay as the video viewer (`apply_normalized_throw_detection`) plus a large bottom-right badge (gray `0`, red `1`).

Saved `.npz` arrays: `labels`, `frame_features` `(N, 4)` elbow/wrist normalized xy, `windows` `(N, BUFFER_SIZE, 4)` causal rolling buffers, `sides`, `clip_paths`, `clip_offsets`, `clip_frame_counts`, `buffer_size`.

## GRU training (`throw_detection.trainer`)

`uv run python -m throw_detection.trainer` opens a tkinter UI. Pick a `.npz` from `throw_detection/training_sets/`, adjust hyperparameters (hidden size, layers, dropout, learning rate, batch size, epochs, validation split, seed, positive-class weight), then **Train**. Training runs on a background thread; the log and progress bar update each epoch with train/val loss, accuracy, precision, and recall. **Stop** finishes after the current epoch.

Validation split is **clip-level** (no frame leakage across clips). Frames with missing pose (`sides == -1`) are excluded. Early-window NaNs in `windows` are zeroed before feeding the GRU.

When training finishes (or is stopped), name the model and **Save** to `throw_detection/models/<name>.pt`. Checkpoints include `state_dict`, model config, source set name, `buffer_size`, training hyperparameters, and metrics/history.

## GRU inference (`throw_detection.inference`)

`ThrowInference(model_path)` loads a saved checkpoint and runs pose → normalized elbow/wrist features → causal rolling window → GRU logit on each frame. `predict(frame, warmup_frames=...)` returns a `ThrowPrediction` (`label`, `logit`, `probability`, `has_pose`, `detection`). Early window slots are zero-filled (matching training). Missing pose yields label `0` with zero logit.

The video viewer **GRU throw inference** filter (`FilterId.GRU_THROW_INFERENCE`) uses the most recently modified `.pt` in `throw_detection/models/` (`video_viewer/config.py` → `THROW_MODEL_PATH`). Overlay: normalized pose, logit/probability readout, and bottom-right `0`/`1` badge (same colors as the labeller). During sequential forward playback, `ThrowInference` streams one new frame per step. On seeks, backward steps, filter changes, or other non-sequential jumps, the viewer rebuilds the rolling buffer from prior frames (YOLO on up to `buffer_size − 1` warmup frames plus the current frame).

## Trajectory tracking (`trajectory_tracking`)

`TrajectoryTracker` is a three-phase state machine called once per frame:

1. **DETECTING_THROW** — waits for `throw_label == 1` from the GRU inference. When detected, moves to phase 2 using the wrist position as the sector origin.
2. **SCANNING_BALL** — on every frame, re-anchors the sector at the wrist if the throw label is still 1. Searches the motion mask for the largest circular contour whose centroid lies inside a circular sector: `sector_radius` pixels from the wrist, centered on the elbow→wrist arm direction, ±`sector_half_angle` degrees wide. When a contour is found, transitions to phase 3.
3. **TRACKING_BALL** — records ball centroid positions. Each frame the sector is re-centered on the last detection and the direction is updated to the previous→current ball vector. If `timeout_frames` consecutive frames yield no detection, the trajectory is finalised: `numpy.polyfit` fits a degree-2 polynomial (y=f(x) or x=f(y) depending on aspect ratio) and 120 sampled curve points are stored. The tracker then returns to phase 1.

A new throw label while in phase 3 immediately finalises the current trajectory and re-enters phase 2.

**Display (`FilterId.TRAJECTORY_TRACKING`):** renders all GRU inference overlays (pose skeleton, logit readout, label badge) plus:
- Sector wedge outline (yellow-orange in phase 2, green in phase 3) at the current scan origin.
- Orange dot at each frame's detected ball position.
- Small teal dots for active trajectory points while tracking.
- Completed trajectory: small purple dots + magenta parabola curve, shown until another valid trajectory is finalised.
- Phase label text in the top-left corner.
- After a throw is fully tracked: speed readout in the top-right (`X.X m/s  Y.Y km/h`), inferred from fitted curve length × torso scale (50 cm assumed shoulder→hip, 10-frame rolling mean) ÷ tracking frame count at the video file's FPS (playback mode only).

**Stereo viewer (`FilterId.STEREO_TRACKING`, stereo viewer only):** left (main) camera runs GRU throw inference (pose skeleton, logit, label badge) plus wrist-anchored ball tracking. The right (secondary) camera runs ball tracking only, driven by the main throw label (full-frame scan, then sector follow). Both panels show trajectory overlays. Throw detection is not run on the secondary camera. Trajectories with fewer than `MIN_TRAJECTORY_POINTS` detected positions are discarded so brief GRU flickers do not replace a completed throw. A camera that finishes tracking waits in `awaiting_partner` until the other camera also ends before both return to `detecting_throw`.

Tune via `trajectory_tracking/config.py`: `SECTOR_ANGLE_DEG`, `SECTOR_DIRECTION_DEG`, `SECTOR_RADIUS_PX`, `TRACKING_TIMEOUT_FRAMES`, `MIN_TRAJECTORY_POINTS`, `BALL_CIRCULARITY_MIN/MAX`, `ASSUMED_TORSO_CM`, `TORSO_LENGTH_BUFFER_SIZE`.

## Frame sync (`framesync`)

Measures stereo camera desync from a deliberate **sync action**: drop the ball straight down so it bounces on the table. Left camera is **main**.

**Per-camera phases (`CameraSyncTracker`):**

1. **WATCHING** — look for `DROP_STREAK_FRAMES` (default 3) consecutive frames where the ball bbox bottom moves down with little horizontal motion.
2. **SYNCING** — record bottom-of-bbox samples each frame; detect table bounce (macro): vertical velocity sign change down→up, or sharp slowdown (`SLOWDOWN_RATIO`).
3. **CAPTURING** — after bounce on this feed, record `POST_BOUNCE_CAPTURE_FRAMES` (default 3) more samples independently.
4. **DONE** — hold samples until the partner camera also finishes.

**Stereo session (`FrameSyncEngine`):** first camera entering `SYNCING` opens a session; partner must join within `SYNC_PAIRING_WINDOW_FRAMES`. Bounce and capture run on independent per-camera timelines (feeds may be many frames apart). When both reach `DONE`, `estimate_bounce_subframe_index` finds each bounce time to 2 decimal places; `offset = secondary_bounce − main_bounce`. Display: main `+offset`, secondary `−offset`. Most recent offset persists until the next successful sync.

**Display (`FilterId.FRAME_SYNC`, stereo viewer only):** ball detection rectangle, phase label (top-left), large sync label (top-center, `+X.XX` / `−X.XX` or `--`). No pose/GRU dependency.

Tune via `framesync/config.py`: `DROP_STREAK_FRAMES`, `MAX_HORIZONTAL_DELTA_PX`, `POST_BOUNCE_CAPTURE_FRAMES`, `SLOWDOWN_RATIO`, `MIN_DOWNWARD_VY`, `SYNC_TIMEOUT_FRAMES`, `SYNC_PAIRING_WINDOW_FRAMES`, `SYNC_COOLDOWN_SECONDS`.

## Configuration

**`video_viewer/config.py`**

- **Paths:** `RECORDINGS_DIR`, `DEFAULT_VIDEO`
- **Capture:** `TARGET_RECORD_FPS`, `MAX_CAMERA_PROBE`, `DISPLAY_MAX_SIZE`
- **Ball motion:** `DIFF_*`, `FRAME_DIFF_MORPH_KERNEL_SIZE`, `MOG2_*`, `BALL_CIRCULARITY_*`, `BALL_CONTOUR_MIN_AREA`, `DETECTION_RECT_THICKNESS`
- **Pose overlay:** `POSE_JOINT_RADIUS`, `POSE_BONE_THICKNESS`
- **GRU inference:** `THROW_MODEL_PATH` (latest `throw_detection/models/*.pt`)

**`stereo_viewer/config.py`**

- **Paths:** `RECORDINGS_DIR`, `LEFT_VIDEO`, `RIGHT_VIDEO`
- **Display:** `STEREO_DISPLAY_MAX_SIZE` (reuses `video_viewer` max size; each panel gets half width)
- **Capture:** `TARGET_FPS` (alias of `TARGET_RECORD_FPS`)

**`pose_detection/config.py`**

- **Model:** `POSE_MODEL_PATH`, `POSE_DEVICE` (default `"mps"`), `POSE_CONF_THRESHOLD`, `POSE_KEYPOINT_MIN_CONF`

**`training_recorder/config.py`**

- **Paths:** `RECORDINGS_DIR` (repo root `recordings/`)
- **Capture:** `TARGET_RECORD_FPS`, `MAX_CAMERA_PROBE`, `DISPLAY_MAX_SIZE`

**`throw_detection/config.py`**

- **Paths:** `TRAINING_SETS_DIR`, `MODELS_DIR`, `REPO_ROOT`
- **GRU input:** `BUFFER_SIZE` — rolling window length for normalized elbow/wrist features

**`trajectory_tracking/config.py`**

- **Sector:** `SECTOR_ANGLE_DEG` (full angular width, default 150°), `SECTOR_DIRECTION_DEG` (sector center direction, default 135° = left tilted 45° downward), `SECTOR_RADIUS_PX` (max search distance in pixels, default 400)
- **Tracking:** `TRACKING_TIMEOUT_FRAMES` (consecutive miss frames before trajectory is finalised, default 3)
- **Circularity:** `BALL_CIRCULARITY_MIN / MAX` (same defaults as ball detection: 0.5–1.0)
- **Minimum area:** `BALL_CONTOUR_MIN_AREA` (default 100 px²)
- **Minimum length:** `MIN_TRAJECTORY_POINTS` (default 5) — shorter tracks are discarded
- **Speed:** `ASSUMED_TORSO_CM` (default 50), `TORSO_LENGTH_BUFFER_SIZE` (default 10)

**`framesync/config.py`**

- **Drop:** `DROP_STREAK_FRAMES` (default 3), `MAX_HORIZONTAL_DELTA_PX` (default 8)
- **Bounce capture:** `POST_BOUNCE_CAPTURE_FRAMES` (default 3), `SLOWDOWN_RATIO` (default 0.35), `MIN_DOWNWARD_VY` (default 2.0 px/frame)
- **Session:** `SYNC_TIMEOUT_FRAMES`, `SYNC_PAIRING_WINDOW_FRAMES` (default 90 each), `SYNC_COOLDOWN_SECONDS` (default 3.0 — no new detection until cooldown elapses after a successful sync)

## Conventions

- Package code lives under `video_viewer/`, `stereo_viewer/`, `training_recorder/`, `pose_detection/`, `throw_detection/`, `trajectory_tracking/`, and `framesync/`; keep detection logic separate from UI.
- Filters affect preview only unless explicitly designed to process recordings.
- Use `uv` for dependency changes (`uv add <package>`).
- Recorded videos and `.pt` model weights are gitignored.
- Prefer extending existing filter/detection modules over duplicating CV logic in `app.py`.

## Planned direction (not yet built)

- Ball position fusion across views → 3D trajectory
- 3D map UI showing historical throws

When implementing these, update this file and `README.md` to reflect new modules and workflows.
