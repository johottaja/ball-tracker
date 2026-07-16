# balltracker

A small Python app for recording webcam video and inspecting it frame by frame, with a computer-vision pipeline aimed at finding a moving ball in the scene.

The main UI lives in `video_viewer/`: record from a camera, play back clips, step through frames, and apply filters to the live preview or playback.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or another way to install dependencies from `pyproject.toml`
- A webcam (for recording); macOS may prompt for camera permission on first use

## Setup

```bash
cd balltracker
uv sync
```

## Run

**Video viewer** (record, playback, filters):

```bash
uv run python -m video_viewer
```

Or:

```bash
uv run python video_viewer/viewer.py
```

**Training clip recorder** (named training sets, short clips):

```bash
uv run python -m training_recorder
```

## Two-player game tracking

`game_tracker` processes one active throw at a time from either half of the main
(left) camera. The player in the main image's left half is mirrored only in the
GRU's normalized horizontal arm features, allowing the existing right-side throw
model to classify both directions. Its ball-search sector is mirrored in image
space; recorded image coordinates and reconstructed 3D geometry are not.

Calibration automatically determines whether the secondary camera sees the same
player on its left or right by comparing the cameras' calibrated projection
orientations. The game JSON records each accepted throw's `thrower_side`; the
3D visualizer renders main-camera-left throws in blue and preserves red for
right-side and older throws.

Reprocess recordings after upgrading: the pose and GRU cache formats now retain
separate left/right player slots, so older single-player caches are regenerated.

## Using the training recorder

- Enter a **Training set** name (e.g. `throws_v1`). Clips for the same name are saved together.
- Choose a camera and use **Refresh** if needed.
- **Start clip** / **Stop clip** saves each take as `recordings/<training_set>/clip_<timestamp>.mp4` at the repo root.
- This folder is separate from `video_viewer/recordings/`, which the viewer uses for its single default recording.

## Using the viewer

**Record mode**

- Choose a camera from the dropdown and use **Refresh** if you plug one in later.
- **Start recording** / **Stop recording** saves to `video_viewer/recordings/recording.mp4` (raw video, no filters burned in).

**Playback mode**

- Switch to **Playback** after recording, or use **Open video…** to load a file.
- Controls: go to start, step one frame back/forward, play, pause.

**Filters**

- Pick a filter from the **Filter** dropdown (or the **Filters** menu on macOS when the app is focused).
- Filters affect the display only, not the saved recording.
- Several **Diff …** filters expose intermediate steps of the same pipeline (subtract, brightness, threshold, contours) for debugging. The end-to-end ball finder is **Ball detection**.

## Ball detection filter

**Ball detection** is the production-style filter: it runs the full motion pipeline on each frame (using the **previous frame** as reference), finds the best ball-shaped blob, and draws a thick red rectangle on the **original** camera image.

### Pipeline steps

1. **Frame difference** — `cv2.subtract(current, previous)` highlights what changed between consecutive frames. Static background tends to cancel out; a moving ball leaves a bright region.

2. **Brightness amplification** — the diff is scaled with `cv2.convertScaleAbs` (`DIFF_BRIGHTNESS_FACTOR`, default `5.0`) so faint motion is easier to threshold.

3. **Threshold** — the amplified image is converted to grayscale and binarized with `cv2.threshold` (`DIFF_THRESH_VALUE`, default `50`). Pixels above the cutoff become white (255).

4. **Morphological cleaning** — `cv2.morphologyEx(..., MORPH_OPEN)` with a 5×5 kernel removes small noise specks from the binary mask.

5. **Contour detection** — external contours are found on the cleaned mask (`cv2.findContours` with `RETR_EXTERNAL`).

6. **Circularity filter** — for each contour, circularity is  
   `(4π × area) / perimeter²`.  
   Contours in `(BALL_CIRCULARITY_MIN, BALL_CIRCULARITY_MAX]` (defaults `0.5`–`1.0`) are kept. A circle scores ~1.0; elongated or jagged shapes (e.g. arms) score lower and are rejected.

7. **Largest ball** — among circular contours, the one with the largest area is treated as the ball.

8. **Overlay** — a red bounding box (`cv2.boundingRect`) is drawn on the unmodified input frame (`DETECTION_RECT_THICKNESS`, default `6`).

### Tuning

Edit `video_viewer/config.py`:

| Parameter | Role |
|-----------|------|
| `DIFF_BRIGHTNESS_FACTOR` | How much to boost the frame diff before threshold |
| `DIFF_THRESH_VALUE` | Binary cutoff on amplified motion |
| `MORPH_KERNEL_SIZE` | Open kernel size for noise removal |
| `BALL_CIRCULARITY_MIN` / `BALL_CIRCULARITY_MAX` | Acceptable circularity range for a ball |
| `DETECTION_RECT_THICKNESS` | Box thickness on the output frame |
| `TARGET_RECORD_FPS` | Requested capture FPS (minimum enforced in software) |

Detection works best when the ball moves clearly between frames and the background is relatively stable. The first frame after a seek or camera switch has no valid previous frame, so nothing is detected until the next frame.

## Project layout

```
balltracker/
├── pyproject.toml          # dependencies (OpenCV, Pillow)
├── recordings/             # training clips by set name (gitignored)
├── training_recorder/      # training clip recorder GUI
├── video_viewer/
│   ├── __main__.py         # entry point
│   ├── app.py              # tkinter UI
│   ├── camera.py           # webcam open / probe / FPS
│   ├── config.py           # paths and tuning constants
│   ├── display.py          # resize frames for the UI
│   ├── filters.py          # filter pipeline and stages
│   ├── ball_detection.py   # contours, circularity, drawing
│   ├── recording.py          # VideoWriter helper
│   └── recordings/         # default save location (gitignored)
└── main.py                 # placeholder; use video_viewer to run
```

## Dependencies

- **OpenCV** — capture, read/write video, image processing
- **Pillow** — convert frames for display in tkinter
- **tkinter** — GUI (stdlib)
