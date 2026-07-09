# AGENTS.md

Guidance for AI agents working in this repository.

**Keep this file up to date.** When you add modules, change architecture, introduce dependencies, or shift project goals, update AGENTS.md in the same change so future agents have accurate context.

## Project overview

**balltracker** is a beer pong throw tracker. The long-term goal is to record a table from two cameras at different angles, track each ball’s trajectory through the throw, and display all throws on a 3D map.

**Current state:** Six Python desktop apps plus shared detection libraries:

- **`video_viewer/`** — record webcam video and inspect it frame by frame. Includes configurable **ball detection** (MOG2 + morphological closing, or frame diff → threshold) with contour/circularity filtering, plus **throw detection** (YOLOv11 pose overlay via `pose_detection`).
- **`stereo_viewer/`** — dual-camera version of the video viewer: side-by-side live preview and playback, same filter set applied independently per camera (plus **Stereo tracking** and **Frame sync**, stereo-only). Records `left.mp4` and `right.mp4` under `stereo_viewer/recordings/`; on stop saves per-camera capture timestamps to `stereo_timeline.json` and aligns playback on a shared master timeline (no video re-encode). Playback **Preprocess** runs batched YOLO pose inference then batched GRU throw inference (`yolo_inferences.npz`, `gru_inferences.npz`).
- **`game_tracker/`** — production dual-camera app for recording a full beer pong game. No debug filters; always runs stereo throw + ball tracking, triangulates 3D trajectories from configurable camera geometry, and saves throws to `game_tracker/games/*.json` for a future React SPA. Records native `left.mp4` / `right.mp4` under `game_tracker/recordings/` plus `stereo_timeline.json` for time-domain stereo alignment (same module as `stereo_viewer`).
- **`pose_detection/`** — reusable YOLO pose pipeline: per-frame dominant-hand selection and batch extraction of arm keypoints from frame sequences.
- **`training_recorder/`** — lightweight GUI for recording labeled training clips. Enter a training set name; each clip is saved under `recordings/<training_set>/` at the repo root (separate from `video_viewer/recordings/`).
- **`throw_detection/`** — throw-event labeling GUI, GRU training-data export, GRU training GUI, and streaming GRU inference. Labels per-frame throw/not-throw on clips from `recordings/<set>/`; saves NumPy `.npz` datasets under `throw_detection/training_sets/`; trained models under `throw_detection/models/`.
- **`trajectory_tracking/`** — stateful ball trajectory tracker that combines throw inference with configurable ball motion masks. Three phases: detecting throw → scanning for ball in a circular sector from the wrist → tracking ball frame-by-frame. Fits a parabola to the collected positions and exposes drawing helpers for the video viewer filter.
- **`framesync/`** — stereo camera frame-offset measurement from deliberate straight-down ball drops and table bounces. Per-camera macro phase machine plus subframe bounce-time estimation; reused by the stereo viewer **Frame sync** filter.
- **`calibration/`** — shared table-corner calibration UI and homography math. **Calibrate** in `stereo_viewer` and `game_tracker` saves `calibration.json` at the repo root (gitignored): table dimensions, calibration frame size, per-camera 3×4 projection matrices (computed from corner clicks at save time), and persisted camera layout stats (positions, angles, FOVs, stereo baseline). The dialog can pin the right camera’s focal length to the left (for digitally cropped/zoomed feeds such as a vertical iPhone forced into 16:9) or accept an explicit right horizontal FOV. `game_tracker` triangulates 3D throws directly from the saved projection matrices via `cv2.triangulatePoints`; **Camera layout** reads the saved layout stats.

Dual-camera synchronized recording is available via `stereo_viewer` and `game_tracker`. Stereo 3D triangulation and JSON export are implemented in `game_tracker`; a React 3D map UI is not built yet.

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
uv run python -m game_tracker
uv run python -m training_recorder
uv run python -m throw_detection.labeller <set_name>
uv run python -m throw_detection.trainer
```

Alternative entry: `uv run python video_viewer/viewer.py`

`main.py` at the repo root is a placeholder; use `video_viewer`, `stereo_viewer`, `game_tracker`, `training_recorder`, `throw_detection.labeller`, or `throw_detection.trainer` to run an app.

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
│   ├── config.py             # sector, tracking, torso/speed, release-backtrack constants
│   ├── speed.py              # TorsoLengthBuffer, curve-length speed estimate
│   ├── release.py            # palm estimate, parabola backtrack to release point
│   ├── tracker.py            # Phase enum, TrajectoryResult, TrajectoryTracker
│   └── drawing.py            # draw_trajectory_overlay (sector, points, parabola, speed)
├── framesync/                # Stereo frame-offset from ball drop/bounce
│   ├── __init__.py
│   ├── config.py             # drop/bounce thresholds, session timeouts
│   ├── types.py              # Phase, BallSample, FrameSyncResult
│   ├── tracker.py            # CameraSyncTracker (per-camera state machine)
│   ├── engine.py             # FrameSyncEngine (stereo session + offset math)
│   ├── playback.py           # Seek reset + sync-event cache helpers for playback
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
├── game_tracker/             # Production game recorder + 3D throw tracking
│   ├── __init__.py
│   ├── __main__.py           # `python -m game_tracker` entry
│   ├── app.py                # GameTrackerApp — record/playback, ball method, calibration
│   ├── config.py             # Paths, triangulation thresholds
│   ├── display.py            # Re-exports stereo_viewer display helpers
│   ├── processor.py          # GameTrackingProcessor — stereo tracking + JSON export
│   ├── triangulation.py      # Projection-matrix triangulation, 3D curve fit
│   ├── game_data.py          # GameSession / ThrowRecord JSON schema
│   └── recordings/           # left.mp4, right.mp4, game.json (gitignored)
├── calibration/              # Table-corner calibration UI + homography storage
│   ├── __init__.py
│   ├── config.py             # CALIBRATION_JSON path
│   ├── types.py              # TableCalibration, CameraCalibration, layout stats types
│   ├── homography.py         # corner→H, H→projection matrix, triangulation helpers
│   ├── layout.py             # camera layout stats from projection matrices
│   ├── layout_dialog.py      # Camera layout top-down visualization
│   ├── storage.py            # load/save calibration.json (+ layout attach/migrate)
│   ├── dialog.py             # TableCalibrationDialog
│   └── frames.py             # capture_stereo_pair from record/playback state
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
    ├── prefetch.py           # 1-frame playback lookahead (worker thread + filter state sync)
    ├── recording.py          # VideoWriter helper
    ├── filters.py            # Filter registry and FrameFilter pipeline
    ├── ball_motion.py        # BallDetectionMethod, MotionMaskBuilder (MOG2 / frame diff)
    ├── ball_detection.py     # Contour/circularity logic and ball overlays
    ├── stereo_ball_detection.py # Shared stereo mask + full-frame ball-bottom detection
    ├── pose_overlay.py       # Dominant-hand skeleton overlay for the viewer filter
    ├── yolo_batch.py         # Batched YOLO pose inference + npz cache
    ├── gru_batch.py          # Batched GRU throw inference from pose cache + npz cache
    ├── pose_estimation.py    # Playback Preprocess panel (YOLO ± GRU)
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
| `stereo_viewer/app.py` | `StereoViewerApp` — two camera streams, side-by-side preview/playback, independent `FrameFilter` per camera (or coordinated **Stereo tracking**); extends the shorter recording on stop using capture timestamps; **Import from game tracker** copies `game_tracker/recordings/left.mp4` and `right.mp4` into stereo viewer recordings |
| `stereo_viewer/config.py` | `RECORDINGS_DIR`, `LEFT_VIDEO`, `RIGHT_VIDEO`, `STEREO_DISPLAY_MAX_SIZE` |
| `stereo_viewer/display.py` | `panel_size_for_frame`, `stereo_frame_to_photo` (horizontal composite) |
| `stereo_viewer/stereo_tracking.py` | `StereoTrackingProcessor` — main GRU + ball track on both cameras; secondary ball-only track |
| `stereo_viewer/frame_sync.py` | `FrameSyncProcessor` — ball drop/bounce sync on both cameras via `FrameSyncEngine` |
| **game_tracker** | |
| `game_tracker/app.py` | `GameTrackerApp` — dual-camera record/playback, ball-detection method, table calibration; **Import from stereo viewer** copies `stereo_viewer/recordings/left.mp4` and `right.mp4` into game tracker recordings |
| `game_tracker/config.py` | `RECORDINGS_DIR`, `LEFT_VIDEO`, `RIGHT_VIDEO`, `GAME_JSON` |
| `game_tracker/processor.py` | `GameTrackingProcessor` — stereo GRU + ball tracking, frame-indexed 2D capture, triangulation, incremental `game.json` writes |
| `game_tracker/triangulation.py` | `cv2.triangulatePoints` from calibration projection matrices; quadratic 3D curve fit; speed from 3D arc length |
| `game_tracker/game_data.py` | `GameSession`, `ThrowRecord`, JSON save/load (atomic temp + rename) |
| **video_viewer** | |
| `app.py` | `VideoViewerApp` — modes (record/playback), UI, frame stepping, filter wiring |
| `filter_controls.py` | `FilterControls` — filter combobox, ball-detection method combobox, Filters menu (both viewers) |
| `playback.py` | Seek helpers, motion-mask/GRU warmup context, `frame_to_display_photo` |
| `prefetch.py` | `PlaybackPrefetcher` — background filter apply for next frame during forward play |
| `camera.py` | Open cameras (AVFoundation on macOS), probe indices, enforce min FPS; `CameraReader` captures on a background thread |
| `config.py` | `RECORDINGS_DIR`, ball-motion thresholds (MOG2, frame diff), pose overlay drawing sizes |
| `filters.py` | `FilterId` enum, `FrameFilter` state |
| `ball_motion.py` | `BallDetectionMethod`, `MotionMaskBuilder` — MOG2, frame diff, hybrid, and hybrid-stacked masks |
| `ball_detection.py` | Circular contour filtering, largest-ball selection, `contour_bottom_center`, drawing |
| `pose_overlay.py` | Throw / normalized-throw / GRU-inference filter overlays (imports `pose_detection`, `throw_detection.inference`) |
| `recording.py` | Create MP4 writer; legacy `extend_video_to_reference` / `extend_video_evenly` helpers (no longer used by stereo apps) |
| `stereo_timeline.py` | `StereoTimeline` — master playback slots from per-camera `time.monotonic()` capture timestamps; save/load `stereo_timeline.json` |
| `stereo_playback.py` | `StereoFrameReader` — read native source frames through master timeline; native ±1 neighbors for frame diff on hold slots |
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
| `config.py` | `SECTOR_ANGLE_DEG`, `SECTOR_DIRECTION_DEG`, `SECTOR_RADIUS_PX`, `TRACKING_TIMEOUT_FRAMES`, `BALL_CIRCULARITY_MIN/MAX` (0.4–1.0), `ASSUMED_TORSO_CM`, `TORSO_LENGTH_BUFFER_SIZE` |
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

**Hybrid:** runs MOG2 and frame diff independently; each does its own contour detection. Ball position merges at the detection level — if both find a ball in the search area, MOG2 wins; otherwise either method’s hit is used. Used by trajectory/stereo tracking via `alternate_motion_mask`. **Contours** draws circular contours from both masks on an OR background.

**Hybrid stacked:** bitwise-OR (`cv2.bitwise_or`) of the MOG2 and frame-diff masks, then standard single-mask contour detection. Behaves like one combined motion mask everywhere.

Shared contour step (all methods):

1. **Contour detection** — external contours on the binary mask
2. **Circularity filter** — reject non-ball shapes via `BALL_CIRCULARITY_MIN/MAX`
3. **Minimum area** — reject contours smaller than `BALL_CONTOUR_MIN_AREA` px²
4. **Largest contour** — treated as the ball

**Contours** filter draws all circular contours on the mask. **Ball detection** draws a red bounding rectangle on the original frame.

Filters are display-only; recordings save raw camera frames.

## Pose / throw detection

`pose_detection/` loads `yolo11n-pose.pt` on first use. Frames are downscaled so the longest side is at most `POSE_INFERENCE_MAX_SIZE` (default 640) before inference; detected keypoints are mapped back to full-resolution coordinates. For each frame it detects people, evaluates left/right arm chains (COCO keypoints 5–10), and picks the wrist closest to the frame center as the “dominant hand.”

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
2. **SCANNING_BALL** — on every frame while the throw label is 1, re-anchors the sector at the wrist and pauses the scan timer. Searches the motion mask for the largest circular contour whose centroid lies inside a circular sector: `sector_radius` pixels from the wrist, centered on the elbow→wrist arm direction, ±`sector_half_angle` degrees wide. When a contour is found, transitions to phase 3. After the throw label returns to 0, exits if no ball was found within `SCANNING_TIMEOUT_FRAMES` (stereo: follow partner; mono: `DETECTING_THROW`).
3. **TRACKING_BALL** — records ball centroid positions. Each frame the sector is re-centered on the last detection and the direction is updated to the previous→current ball vector. If `timeout_frames` consecutive frames yield no detection, the trajectory is finalised: `numpy.polyfit` fits a degree-2 polynomial (y=f(x) or x=f(y) depending on aspect ratio) and 120 sampled curve points are stored. The tracker then returns to phase 1.

A new throw label while in phase 3 immediately finalises the current trajectory and re-enters phase 2.

**Release backtrack (after finalize):** On the main camera, the fitted parabola is walked backward frame-by-frame through cached YOLO pose. Palm position is estimated as `elbow + PALM_EXTENSION × (wrist − elbow)` (default 1.2). The frame with minimum trajectory–palm distance within the GRU throw window is prepended as the release point. The secondary camera extrapolates its parabola to the same frame. Used in `game_tracker` JSON export and stereo/mono playback overlays when pose cache is available.

**Display (`FilterId.TRAJECTORY_TRACKING`):** renders all GRU inference overlays (pose skeleton, logit readout, label badge) plus:
- Sector wedge outline (yellow-orange in phase 2, green in phase 3) at the current scan origin.
- Orange dot at each frame's detected ball position.
- Small teal dots for active trajectory points while tracking.
- Completed trajectory: small purple dots + magenta parabola curve, shown until another valid trajectory is finalised.
- Phase label text in the top-left corner.
- After a throw is fully tracked: speed readout in the top-right (`X.X m/s  Y.Y km/h`), inferred from fitted curve length × torso scale (50 cm assumed shoulder→hip, 10-frame rolling mean) ÷ tracking frame count at the video file's FPS (playback mode only).

**Stereo viewer (`FilterId.STEREO_TRACKING`, stereo viewer only):** left (main) camera runs GRU throw inference (pose skeleton, logit, label badge) plus wrist-anchored ball tracking. The right (secondary) camera runs ball tracking only, driven by the main throw label; during **scanning ball** it uses right-camera YOLO pose (from preprocess cache or live inference) to search the same wrist-anchored sector as the main feed, falling back to full-frame scan when pose is missing. Both panels show trajectory overlays plus framesync overlay (phase label, ±offset badge). Ball motion masks and full-frame ball-bottom detection are shared with the framesync engine via `video_viewer.stereo_ball_detection` (no duplicate mask work). GRU throw detection is not run on the secondary camera. Trajectories with fewer than `MIN_TRAJECTORY_POINTS` detected positions are discarded so brief GRU flickers do not replace a completed throw. A failed camera (discarded trajectory or scan timeout) immediately adopts the partner's phase instead of blocking in `awaiting_partner`. Valid completions wait in `awaiting_partner` until the partner also completes or `AWAITING_PARTNER_TIMEOUT_FRAMES` elapses; when both are awaiting on the same frame, both return to `detecting_throw` immediately.

Tune via `trajectory_tracking/config.py`: `SECTOR_ANGLE_DEG`, `SECTOR_DIRECTION_DEG`, `SECTOR_RADIUS_PX`, `TRACKING_TIMEOUT_FRAMES`, `MIN_TRAJECTORY_POINTS`, `BALL_CIRCULARITY_MIN/MAX`, `ASSUMED_TORSO_CM`, `TORSO_LENGTH_BUFFER_SIZE`.

## Game tracker (`game_tracker`)

Production app for recording a beer pong session and exporting 3D throw trajectories to JSON.

**UI:** Same record/playback/camera-selection shell as `stereo_viewer`, but **no display filters**. Ball detection method combobox on playback only (MOG2 + closing, frame diff, hybrid, or hybrid stacked). **Calibrate** opens the shared table-corner calibration dialog (saves `calibration.json` at repo root).

**Playback overlays:** Same stereo tracking visualization as `stereo_viewer` **Stereo tracking** (left: GRU pose + logit + label badge + trajectory overlay; right: trajectory overlay only) — no framesync. Loads `yolo_inferences.npz` / `gru_inferences.npz` from `game_tracker/recordings/` into `PlaybackCache` when present (written by **Process game…**).

**Recording:** Raw frames to `game_tracker/recordings/left.mp4` and `right.mp4` at native frame counts. On stop, per-camera capture timestamps are saved to `stereo_timeline.json`; master timeline length equals the longer clip. Playback and batch processing read aligned frames through `StereoFrameReader` (holds on the lagging side only — no duplicate frames baked into MP4s).

**Offline processing (`batch_process.process_game_recording`):** three phases — batched YOLO pose (`yolo_inferences.npz`), batched GRU throw labels on the main camera from pose features (`gru_inferences.npz`, keyed to the active `.pt` model), then ball/trajectory tracking with both caches loaded into `PlaybackCache`.

**Tracking (`GameTrackingProcessor`):** Active during batch processing after record. Main (left) camera: GRU throw inference + wrist-anchored ball tracking. Secondary (right): ball tracking driven by main throw label, with wrist-anchored sector scan on the right feed during **scanning ball** (right YOLO from preprocess cache or live inference; full-frame fallback when pose is missing). Stereo phase gate (`AWAITING_PARTNER`) pairs valid completions across cameras; failed tracks adopt the partner phase via `trajectory_tracking.stereo.reconcile_stereo_trackers`. Frame-indexed 2D detections captured during `TRACKING_BALL` on the master timeline; when both cameras complete a throw, `triangulate_throw` pairs left/right by master capture time from `stereo_timeline.json` (linear interpolation on the secondary track). Speed uses actual slot durations from the timeline, not nominal FPS.

**3D coordinate system:** Origin at table center; X along table length, Y along width, Z up from table (meters). **Calibrate** saves 3×4 projection matrices per camera (derived from corner clicks + focal-length estimation at save time). Per-point triangulation via `cv2.triangulatePoints` on temporally aligned 2D pairs (`secondary_frame = main_frame + offset` when offset is known). Video frame size must match the calibration frame size. 3D speed from fitted curve arc length ÷ throw duration. Requires `calibration.json` at repo root (saved via **Calibrate**).

**`game.json` schema (version 1):** `recorded_at`, `fps`, `frame_count`, `videos`, `coordinate_system`, optional `calibration` (full `calibration.json` snapshot including `layout` stats, written at process time), `throws[]` with `id`, `start_frame`, `end_frame`, `points_3d`, `fitted_curve_3d`, `speed_m_s`, `tracks_2d` (left/right pixel tracks). Designed for consumption by a future React SPA.

## Frame sync (`framesync`)

Measures stereo camera desync from a deliberate **sync action**: drop the ball straight down so it bounces on the table. Left camera is **main**.

**Per-camera phases (`CameraSyncTracker`):**

1. **WATCHING** — look for `DROP_STREAK_FRAMES` (default 3) consecutive frames where the ball bbox bottom moves down with little horizontal motion.
2. **SYNCING** — record bottom-of-bbox samples each frame; detect table bounce (macro): vertical velocity sign change down→up, or sharp slowdown (`SLOWDOWN_RATIO`).
3. **CAPTURING** — after bounce on this feed, record `POST_BOUNCE_CAPTURE_FRAMES` (default 3) more samples independently.
4. **DONE** — hold samples until the partner camera also finishes.

**Stereo session (`FrameSyncEngine`):** first camera entering `SYNCING` opens a session; partner must join within `SYNC_PAIRING_WINDOW_FRAMES`. Bounce and capture run on independent per-camera timelines (feeds may be many frames apart). When both reach `DONE`, `estimate_bounce_subframe_index` finds each bounce time to 2 decimal places; `offset = secondary_bounce − main_bounce`. Display: main `+offset`, secondary `−offset`. Most recent offset persists until the next successful sync. On playback seeks, `framesync.playback` restores the latest cached offset from `PlaybackCache` sync events.

**Display (`FilterId.FRAME_SYNC`, stereo viewer only):** ball detection rectangle, phase label (top-left), large sync label (top-center, `+X.XX` / `−X.XX` or `--`). No pose/GRU dependency. Uses the same `video_viewer.stereo_ball_detection` helper as **Stereo tracking**.

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

**`game_tracker/config.py`**

- **Paths:** `RECORDINGS_DIR`, `LEFT_VIDEO`, `RIGHT_VIDEO`, `GAME_JSON`
- **Triangulation:** `MIN_TRIANGULATION_HEIGHT_M`, `MAX_TRIANGULATION_HEIGHT_M`, `MAX_TRIANGULATION_RESIDUAL_M`

**`pose_detection/config.py`**

- **Model:** `POSE_MODEL_PATH`, `POSE_DEVICE` (default `"mps"`), `POSE_CONF_THRESHOLD`, `POSE_KEYPOINT_MIN_CONF`, `POSE_INFERENCE_MAX_SIZE` (default `640` — longest side sent to YOLO; keypoints scaled back to full frame)

**`training_recorder/config.py`**

- **Paths:** `RECORDINGS_DIR` (repo root `recordings/`)
- **Capture:** `TARGET_RECORD_FPS`, `MAX_CAMERA_PROBE`, `DISPLAY_MAX_SIZE`

**`throw_detection/config.py`**

- **Paths:** `TRAINING_SETS_DIR`, `MODELS_DIR`, `REPO_ROOT`
- **GRU input:** `BUFFER_SIZE` — rolling window length for normalized elbow/wrist features

**`trajectory_tracking/config.py`**

- **Sector:** `SECTOR_ANGLE_DEG` (full angular width, default 150°), `SECTOR_DIRECTION_DEG` (sector center direction, default 135° = left tilted 45° downward), `SECTOR_RADIUS_PX` (max search distance in pixels, default 400)
- **Tracking:** `TRACKING_TIMEOUT_FRAMES` (consecutive miss frames before trajectory is finalised, default 5)
- **Scan timeout:** `SCANNING_TIMEOUT_FRAMES` (default 10) — consecutive frames in `SCANNING_BALL` with throw label 0 and no ball detection
- **Partner wait:** `AWAITING_PARTNER_TIMEOUT_FRAMES` (default 10) — max frames in `AWAITING_PARTNER` before returning to idle (incl. discarded throws)
- **Circularity:** `BALL_CIRCULARITY_MIN / MAX` in `trajectory_tracking/config.py` (defaults 0.4–1.0; re-exported from `video_viewer/config.py` for ball detection)
- **Minimum area:** `BALL_CONTOUR_MIN_AREA` (default 100 px²)
- **Minimum length:** `MIN_TRAJECTORY_POINTS` (default 5) — shorter tracks are discarded
- **Speed:** `ASSUMED_TORSO_CM` (default 50), `TORSO_LENGTH_BUFFER_SIZE` (default 10)
- **Release backtrack:** `PALM_EXTENSION` (default 1.2), `RELEASE_MAX_LOOKBACK_FRAMES` (default 45), `RELEASE_HIT_RADIUS_FACTOR` (default 0.35 × forearm length)

**`calibration/config.py`**

- **Paths:** `CALIBRATION_JSON` (repo root `calibration.json`, gitignored)

**`calibration.json` schema:** `table_length_m`, `table_width_m`, `image_width`, `image_height`, `cameras[]` with `name` (`left` / `right`) and `projection_matrix` (3×4 nested list, world XYZ → image pixels), and `layout` with per-camera stats (`center`, `xy_distance_m`, `z_m`, `yaw_deg`, `pitch_deg`, `horizontal_fov_deg`, `fov_left_xy`, `fov_right_xy`) plus optional `stereo` (`baseline_xy_m`, `baseline_3d_m`, `delta_z_m`). Layout stats are computed on save; legacy files without `layout` are auto-migrated on load. Corner clicks must be in canonical order (clockwise from above): `(+L/2,+W/2)`, `(+L/2,−W/2)`, `(−L/2,−W/2)`, `(−L/2,+W/2)`.

**`framesync/config.py`**

- **Drop:** `DROP_STREAK_FRAMES` (default 3), `MAX_HORIZONTAL_DELTA_PX` (default 8)
- **Bounce capture:** `POST_BOUNCE_CAPTURE_FRAMES` (default 3), `SLOWDOWN_RATIO` (default 0.35), `MIN_DOWNWARD_VY` (default 2.0 px/frame)
- **Session:** `SYNC_TIMEOUT_FRAMES`, `SYNC_PAIRING_WINDOW_FRAMES` (default 90 each), `SYNC_COOLDOWN_SECONDS` (default 3.0 — no new detection until cooldown elapses after a successful sync)

## Conventions

- Package code lives under `video_viewer/`, `stereo_viewer/`, `game_tracker/`, `calibration/`, `training_recorder/`, `pose_detection/`, `throw_detection/`, `trajectory_tracking/`, and `framesync/`; keep detection logic separate from UI.
- Filters affect preview only unless explicitly designed to process recordings.
- Use `uv` for dependency changes (`uv add <package>`).
- Recorded videos and `.pt` model weights are gitignored.
- Prefer extending existing filter/detection modules over duplicating CV logic in `app.py`.

## Planned direction (not yet built)

- React 3D map UI showing historical throws from `game.json`

When implementing these, update this file and `README.md` to reflect new modules and workflows.
