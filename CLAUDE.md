# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file PoC for detecting children playing with kitchen stoves on home CCTV footage. Uses YOLOv8 pose estimation + a state machine over wrist keypoints + an OpenAI-compatible VLM as a verification layer.

The whole app is `stove_detect_v2.py`. There are no modules, no tests, no build step.

## Run

**Batch Processing (Default)**
```powershell
python stove_detect_v2.py
```
Drop videos into `inputs/`. Outputs land in `output/` (annotated `.avi` per video, `_events.json` per video, `.mp4` clips per flagged event under `output/clips/`). Saved per-video setup lives in `stove_zones.yaml`.

**RTSP/Webcam Live Mode**
Edit `stove_detect_v2.py` and set `LIVE_MODE = True` and `LIVE_SOURCE = "0"` (or an RTSP URL). The script will run continuously and open a live CV2 window.

Dependencies: `ultralytics`, `opencv-python`, `pyyaml`, `numpy`, `openai` (only when `VLM_MOCK_MODE = False`).

## Per-video interactive setup

On first run for each video or live stream, the script opens an OpenCV setup window:

1. **Stove polygon** — left-click points, right-click to undo, `C` to close (≥3 points), `Q` to abort.

Saved to `stove_zones.yaml` as `{stove: [[x,y]...]}`.

To redo a video's setup, delete its key from `stove_zones.yaml`.

## Architecture

Top-to-bottom in `stove_detect_v2.py`:

- **Paths / Config** — `LIVE_MODE`, `LIVE_SOURCE`, `REID_ENABLED`, plus VLM config flags.
- **Constants** — COCO keypoint indices, state→color map, `KPT_CONF_THRESHOLD`, `PROXIMITY_PX`.
- **`compute_appearance`** — Extracts an HSV color histogram of a person bounding box for Re-ID recovery.
- **`classify_child`** — Uses a *Dynamic Stove-Anchor*. It calculates a perspective-invariant ratio by comparing the vertical distance from the person's feet to their shoulders against the distance from their feet to the physical stove top plane. It also uses horizontal ratios as a fallback.
- **Per-wrist state machine** (`tick_wrist`) — `SAFE → APPROACH → TOUCH (≥touch_frames) → DANGER (≥danger_frames)`. 
- **`VLMVerifier`** — Maintains a rolling `pre_buffer` deque. When a window completes, the clip is written and sent to an OpenAI-compatible endpoint. Mock mode skips network calls.
- **HUD + per-person overlay** — Sleek HUD drawing, translucent banner alerts, pill-shaped bounding box labels, and translucent polygon fills.
- **`process_video`** — The main execution loop handling both batch files and live stream processing. Integrates Re-ID matching to recover dropped tracks.

## Important behaviors

- **Re-ID Recovery**: When ByteTrack loses a child (occlusion), the script archives their full state and color histogram into `stale_archive`. If a new ID appears matching the histogram, the state is transferred immediately.
- **Classification stability**: per-track rolling `deque(maxlen=child_stable_frames)` of votes. Locked in after 3 samples, ties broken as CHILD.
- **Wrist invalid frames**: when a wrist keypoint is invalid, the per-wrist tick is *skipped*. Protects in-progress TOUCH/DANGER timers from single-frame pose flickers.
- **VLM clip frames**: the buffer holds the *annotated* frame.

## Git

`.gitignore` excludes `.claude/`, `.omc/`, `CLAUDE.md`, `MEMORY.md`, `memory/`, `ref.txt`, `interact.txt`, `foragent/` and YOLO weights. Only scripts, `.gitignore`, `stove_zones.yaml`, `inputs/`, and `output/` are tracked. Commits should be human-style.
