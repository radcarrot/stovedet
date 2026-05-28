import os
import cv2
import json
import yaml
from pathlib import Path
import time
import base64
import ctypes
import math
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque

# Auto-detect GStreamer and DeepStream Python Bindings
DEEPSTREAM_AVAILABLE = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    import pyds
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    pass

# =========================================================
# DEEPSTREAM CONFIG (NVIDIA Hardware Acceleration)
# =========================================================
# Auto-enabled if DeepStream bindings are available, but user can override
DEEPSTREAM_ENABLED   = os.getenv("DEEPSTREAM_ENABLED", str(DEEPSTREAM_AVAILABLE)).lower() == 'true'
DS_CONFIG_INFER      = "config_infer_yolo_pose.txt"
DS_CONFIG_TRACKER    = "config_tracker.txt"
DS_CUSTOM_PARSER_SO  = "./nvdsinfer_custom_impl_Yolo_pose/libnvdsinfer_custom_impl_Yolo_pose.so"
DS_MODEL_ENGINE      = "yolov8x-pose.engine"

# =========================================================
# PATHS & DATACENTER MODEL CONFIG
# =========================================================
# Use yolov8x-pose.pt in DeepStream/datacenter mode; fallback to local yolo26l-pose.pt for offline Windows testing
MODEL_PATH  = "yolov8x-pose.pt" if DEEPSTREAM_ENABLED else "yolo26l-pose.pt"
INPUT_DIR   = "inputs"
OUTPUT_DIR  = "output"
CLIPS_DIR   = "output/clips"
ZONES_FILE  = "stove_zones.yaml"


# =========================================================
# VLM CONFIG — OpenAI-compatible endpoint
# =========================================================
VLM_MOCK_MODE = True   # True = no real API calls, returns fake responses
VLM_API_KEY   = os.getenv("VLM_API_KEY",   "sk-mock-1234567890abcdef")
VLM_BASE_URL  = os.getenv("VLM_BASE_URL",  "https://api.openai.com/v1")
VLM_MODEL     = os.getenv("VLM_MODEL",     "gpt-4o-mini")
VLM_PRE_SEC   = 2.5    # seconds of context before event
VLM_POST_SEC  = 2.5    # seconds of context after event
VLM_SAMPLE_N  = 5      # frames sampled from clip for VLM input

# =========================================================
# LIVE STREAM CONFIG
# =========================================================
LIVE_MODE    = False                          # True = live RTSP/webcam, False = batch file processing
LIVE_SOURCE  = os.getenv("LIVE_SOURCE", "0")  # RTSP URL or webcam index ("0", "1", etc.)
LIVE_RECORD  = False                          # Record annotated live stream to output/
LIVE_DISPLAY = True                           # Show cv2.imshow window during live mode

# =========================================================
# RE-ID CONFIG
# =========================================================
REID_ENABLED       = True    # Enable appearance-based Re-ID across track breaks
REID_HIST_BINS     = 16      # Bins per channel for HSV histogram
REID_MATCH_THRESH  = 0.55    # cv2.HISTCMP_CORREL minimum (higher = stricter)
REID_MAX_AGE_SEC   = 3.0     # Max seconds to match against disappeared tracks

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR,  exist_ok=True)

if os.sep in MODEL_PATH and not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

_supported = (".mp4", ".avi", ".mov", ".mkv")
VIDEO_FILES = []
if not LIVE_MODE:
    VIDEO_FILES = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(_supported)
    )
    if not VIDEO_FILES:
        raise FileNotFoundError(f"No videos found in '{INPUT_DIR}/'. Add videos and rerun.")
    print(f"Found {len(VIDEO_FILES)} video(s): {VIDEO_FILES}")
else:
    print(f"Live mode enabled — source: {LIVE_SOURCE}")

# =========================================================
# MODEL
# =========================================================
model = None
if not DEEPSTREAM_ENABLED:
    print(f"\nLoading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
else:
    print(f"\n[DEEPSTREAM] Skipping PyTorch model load — inference handled by TensorRT engine.")

# =========================================================
# CONSTANTS
# =========================================================
KPT_CONF_THRESHOLD = 0.4
PROXIMITY_PX       = 15

NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16

STATE_COLOR = {
    "SAFE":     (0,   200,   0),
    "APPROACH": (0,   200, 200),
    "TOUCH":    (0,   165, 255),
    "DANGER":   (0,     0, 255),
}

# =========================================================
# GEOMETRY
# =========================================================
def is_valid(kpt, conf=None):
    if float(kpt[0]) == 0.0 and float(kpt[1]) == 0.0:
        return False
    if conf is not None and conf < KPT_CONF_THRESHOLD:
        return False
    return True

def is_near_stove(pt, poly, proximity_px=None):
    if proximity_px is None:
        proximity_px = PROXIMITY_PX
    dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)
    return dist >= -proximity_px

def calculate_angle(A, B, C):
    """Angle in degrees at joint B formed by segments BA and BC."""
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    mag = ((BA[0]**2 + BA[1]**2) ** 0.5) * ((BC[0]**2 + BC[1]**2) ** 0.5)
    if mag < 1e-6:
        return None
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

# =========================================================
# RE-ID — Appearance Histogram
# =========================================================
def compute_appearance(frame, bbox, kpts=None, confs=None):
    """Compute normalized HSV histogram(s) for Re-ID matching.
    With keypoints: separate torso (shoulder→hip) and leg (hip→ankle) crops
    concatenated into a single 1D feature vector, eliminating background contamination.
    Falls back to full bounding box when keypoints are unavailable."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    def _region_hist(ry1, ry2):
        ry1, ry2 = max(0, int(ry1)), min(fh, int(ry2))
        if ry2 - ry1 < 5:
            return None
        crop = frame[ry1:ry2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None,
                         [REID_HIST_BINS, REID_HIST_BINS, REID_HIST_BINS],
                         [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        return h.flatten()

    if kpts is not None:
        l_sh = _kpt(kpts, confs, LEFT_SHOULDER)
        r_sh = _kpt(kpts, confs, RIGHT_SHOULDER)
        l_hp = _kpt(kpts, confs, LEFT_HIP)
        r_hp = _kpt(kpts, confs, RIGHT_HIP)
        l_an = _kpt(kpts, confs, LEFT_ANKLE)
        r_an = _kpt(kpts, confs, RIGHT_ANKLE)

        sh_y  = min(p[1] for p in [l_sh, r_sh] if p is not None) if (l_sh or r_sh) else None
        hip_y = max(p[1] for p in [l_hp, r_hp] if p is not None) if (l_hp or r_hp) else None
        an_y  = max(p[1] for p in [l_an, r_an] if p is not None) if (l_an or r_an) else None

        if sh_y is not None and hip_y is not None and an_y is not None:
            torso_h = _region_hist(sh_y, hip_y)
            legs_h  = _region_hist(hip_y, an_y)
            if torso_h is not None and legs_h is not None:
                combined = np.concatenate([torso_h, legs_h]).astype(np.float32)
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined /= norm
                return combined

    # Fallback: full bounding box
    crop = frame[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [REID_HIST_BINS, REID_HIST_BINS, REID_HIST_BINS],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

# =========================================================
# CHILD DETECTION — Dynamic Stove-Anchor (Approach 1)
# =========================================================
def _kpt(kpts, confs, idx):
    """Returns (x, y) if keypoint valid, else None."""
    p = kpts[idx]
    c = confs[idx] if confs is not None else 1.0
    return (float(p[0]), float(p[1])) if is_valid(p, c) else None

def _midpoint(a, b):
    if a is not None and b is not None:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return a if a is not None else b



DEBUG_CLASSIFIER = True   # set False to hide live ratio overlay

def classify_child(kpts, confs, bbox_y1, bbox_y2, frame_height,
                   stove_poly=None, hs_ratios_history=None, return_debug=False):
    """
    PRIMARY: Dynamic Stove-Anchor Calibration (Approach 1)
    Uses the user-defined stove polygon as a physical scale anchor.
    Calculates the perspective-invariant height ratio:
    R = (Y_feet - Y_shoulder) / (Y_feet - Y_stove_top)
    
    Why this is robust:
      - Physically, stove top is at ~90cm height from floor.
      - Adult shoulders are physically at ~140cm (>90cm => R > 1.2).
      - Toddler/Child shoulders are physically at ~75cm (<90cm => R < 1.0).
      - Since the ratio is calculated using vertical differences starting from
        the same ground point (the feet), it is completely perspective-invariant!

    TEMPORAL FILTERING: Anthropometric horizontal ratios.
      - In a single frame, a broad-shouldered adult turning sideways (profile)
        appears to have narrow shoulders, causing head/shoulder ratio to spike,
        triggering false child alerts.
      - By passing `hs_ratios_history` (a deque of recent ratios), we compute
        the *rolling minimum* value, which captures their widest front-facing
        shoulder profile over time.
    """
    # Grab basic keypoints for ratio calculations
    nose   = _kpt(kpts, confs, NOSE)
    l_eye  = _kpt(kpts, confs, LEFT_EYE)
    r_eye  = _kpt(kpts, confs, RIGHT_EYE)
    l_ear  = _kpt(kpts, confs, LEFT_EAR)
    r_ear  = _kpt(kpts, confs, RIGHT_EAR)
    l_sh   = _kpt(kpts, confs, LEFT_SHOULDER)
    r_sh   = _kpt(kpts, confs, RIGHT_SHOULDER)
    l_hip  = _kpt(kpts, confs, LEFT_HIP)
    r_hip  = _kpt(kpts, confs, RIGHT_HIP)
    l_an   = _kpt(kpts, confs, LEFT_ANKLE)
    r_an   = _kpt(kpts, confs, RIGHT_ANKLE)

    sh_mid    = _midpoint(l_sh,  r_sh)
    hip_mid   = _midpoint(l_hip, r_hip)
    ankle_mid = _midpoint(l_an,  r_an)
    l_kn = _kpt(kpts, confs, LEFT_KNEE)
    r_kn = _kpt(kpts, confs, RIGHT_KNEE)

    # Biomechanical crouching detection via joint angles
    is_crouching = False
    _knee_a, _hip_a = [], []
    if l_hip and l_kn and l_an:
        a = calculate_angle(l_hip, l_kn, l_an)
        if a is not None: _knee_a.append(a)
    if r_hip and r_kn and r_an:
        a = calculate_angle(r_hip, r_kn, r_an)
        if a is not None: _knee_a.append(a)
    if l_sh and l_hip and l_kn:
        a = calculate_angle(l_sh, l_hip, l_kn)
        if a is not None: _hip_a.append(a)
    if r_sh and r_hip and r_kn:
        a = calculate_angle(r_sh, r_hip, r_kn)
        if a is not None: _hip_a.append(a)
    if any(a < 130 for a in _knee_a) or any(a < 110 for a in _hip_a):
        is_crouching = True

    # 1. Update temporal ratio buffer if keypoints are present
    head_w = None
    if l_ear is not None and r_ear is not None:
        head_w = abs(r_ear[0] - l_ear[0])
    elif l_eye is not None and r_eye is not None:
        head_w = abs(r_eye[0] - l_eye[0]) * 1.6

    shoulder_w = abs(r_sh[0]  - l_sh[0]) if (l_sh and r_sh) else None
    
    if head_w is not None and shoulder_w and shoulder_w > 5 and hs_ratios_history is not None:
        raw_hs_ratio = head_w / shoulder_w
        hs_ratios_history.append(raw_hs_ratio)

    # ─── Dynamic Stove-Anchor (Primary Approach 1) ───
    if stove_poly is not None and len(stove_poly) >= 3:
        # Estimate ground contact Y (feet)
        y_feet = None
        feet_src = ""
        if l_an and r_an:
            y_feet = (l_an[1] + r_an[1]) / 2.0; feet_src = "ankles"
        elif l_an:
            y_feet = l_an[1]; feet_src = "L_ankle"
        elif r_an:
            y_feet = r_an[1]; feet_src = "R_ankle"
        elif l_kn and r_kn:
            y_feet = (l_kn[1] + r_kn[1]) / 2.0 + 30.0; feet_src = "knees_offset"
        elif l_hip and r_hip:
            y_feet = (l_hip[1] + r_hip[1]) / 2.0 + 80.0; feet_src = "hips_offset"
        else:
            y_feet = float(bbox_y2); feet_src = "bbox_bottom"

        # Determine shoulder midpoint Y
        y_shoulder = None
        sh_src = ""
        if l_sh and r_sh:
            y_shoulder = (l_sh[1] + r_sh[1]) / 2.0; sh_src = "shoulder_mid"
        elif nose:
            y_shoulder = nose[1]; sh_src = "nose"
        elif l_sh:
            y_shoulder = l_sh[1]; sh_src = "L_shoulder"
        elif r_sh:
            y_shoulder = r_sh[1]; sh_src = "R_shoulder"

        # Top of stove polygon represents physical counter height (90cm)
        y_stove_top = float(np.min(stove_poly[:, 1]))

        # Validate that the person is standing in front of or near the stove's depth plane
        # If y_feet <= y_stove_top, they are behind the stove/counter, so we skip this anchor
        if y_feet is not None and y_shoulder is not None and y_feet > y_stove_top:
            denom = y_feet - y_stove_top
            if denom > 5:
                ratio = (y_feet - y_shoulder) / denom
                # Crouching adults can drop ratio below 1.15 — tighten threshold when crouching detected
                child_threshold = 1.05 if is_crouching else 1.15
                is_child = ratio < child_threshold

                if return_debug:
                    return is_child, {
                        "method":        "stove_anchor",
                        "sh_src":        sh_src,
                        "feet_src":      feet_src,
                        "ratio":         ratio,
                        "y_feet":        y_feet,
                        "y_shoulder":    y_shoulder,
                        "y_stove_top":   y_stove_top,
                        "is_crouching":  is_crouching,
                        "result":        is_child,
                    }
                return is_child

    # ─── Fallback Crouch-Robust Classifier (Horizontal Ratios + Temporal Filtering) ───
    hip_w = abs(r_hip[0] - l_hip[0]) if (l_hip and r_hip) else None

    crouching = is_crouching

    debug = {}
    scores = []

    # Apply temporal ratio filtering: take the MINIMUM ratio from history
    # which captures the widest front-facing shoulder span (avoiding side-profile false child classifications).
    active_hs_ratio = None
    if hs_ratios_history and len(hs_ratios_history) > 0:
        active_hs_ratio = min(hs_ratios_history)  # Rolling min
    elif head_w is not None and shoulder_w and shoulder_w > 5:
        active_hs_ratio = head_w / shoulder_w     # Single-frame fallback

    if active_hs_ratio is not None:
        v = active_hs_ratio > 0.47
        scores.append(v); debug["hd/sh"] = (active_hs_ratio, v)

    if shoulder_w and hip_w and hip_w > 5:
        r = shoulder_w / hip_w
        v = r < 1.25
        scores.append(v); debug["sh/hp"] = (r, v)

    if head_w is not None and hip_w and hip_w > 5:
        r = head_w / hip_w
        v = r > 0.50
        scores.append(v); debug["hd/hp"] = (r, v)

    if not crouching:
        bbox_h = max(bbox_y2 - bbox_y1, 1)
        r = bbox_h / frame_height
        v = r < 0.60
        scores.append(v); debug["bbH"] = (r, v)

    debug["method"]      = "horizontal_ratios"
    debug["n_signals"]   = len(scores)
    debug["child_votes"] = sum(scores)
    debug["crouching"]   = crouching

    if len(scores) < 2:
        result = False
    else:
        result = sum(scores) >= len(scores) / 2
    debug["result"] = result

    if return_debug:
        return result, debug
    return result

# =========================================================
# PERSON + WRIST STATE
# =========================================================
def init_person(child_stable_frames):
    return {
        "child_vote":         deque(maxlen=child_stable_frames),
        "hs_ratios_history":  deque(maxlen=30),  # Stores head-to-shoulder ratios over time (1 second at 30fps)
        "is_child":           False,
        "left_wrist":         _init_wrist(),
        "right_wrist":        _init_wrist(),
        "left_wrist_history":  deque(maxlen=3),   # Rolling history for smoothing (3 frames)
        "right_wrist_history": deque(maxlen=3),   # Rolling history for smoothing (3 frames)
        "appearance_hist":    None,               # HSV histogram for Re-ID matching
    }

def _init_wrist():
    return {"frames_in_zone": 0, "state": "SAFE", "last_trigger_frame": -9999}

def tick_wrist(ws, pt, poly, touch_frames, danger_frames, proximity_px=None):
    if is_near_stove(pt, poly, proximity_px):
        ws["frames_in_zone"] += 1
        f = ws["frames_in_zone"]
        if f >= danger_frames:
            ws["state"] = "DANGER"
        elif f >= touch_frames:
            ws["state"] = "TOUCH"
        else:
            ws["state"] = "APPROACH"
    else:
        ws["frames_in_zone"] = 0
        ws["state"]          = "SAFE"
    return ws

# =========================================================
# VLM VERIFIER — OpenAI-compatible clip verification
# =========================================================
class VLMVerifier:
    """
    Maintains rolling pre-event buffer. On trigger(), captures the
    pre-buffer plus next post_frames frames, saves as mp4 clip, and
    asynchronously sends sampled frames to an OpenAI-compatible
    vision-language model for verification.
    """

    SYSTEM_PROMPT = (
        "You are a child-safety verifier reviewing kitchen CCTV clips. "
        "An automated system flagged a possible stove-interaction event. "
        "Your job: confirm whether a CHILD (not an adult) is actually "
        "touching, reaching for, or playing with the stove."
    )

    USER_PROMPT_TEMPLATE = (
        "Event type flagged by detector: {event_type}.\n"
        "{n} frames sampled from a 5-second clip (2.5s before + 2.5s after).\n"
        "Reply with strict JSON only:\n"
        '{{"verified": true|false, "confidence": 0.0-1.0, '
        '"is_child": true|false, "description": "<one short sentence>"}}'
    )

    def __init__(self, fps, video_stem):
        self.fps          = fps
        self.video_stem   = video_stem
        self.pre_frames   = max(1, int(VLM_PRE_SEC  * fps))
        self.post_frames  = max(1, int(VLM_POST_SEC * fps))
        self.pre_buffer   = deque(maxlen=self.pre_frames)
        self.pending      = []   # events waiting for post-frames
        self.results      = []
        self._results_lock = threading.Lock()
        self._threads = []

        self.client = None
        if not VLM_MOCK_MODE:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)
            except ImportError:
                print("[VLM] openai package not installed — forcing mock mode.")
                self.client = None

    def push_frame(self, frame):
        self.pre_buffer.append(frame.copy())
        for p in self.pending:
            if p["post_remaining"] > 0:
                p["post"].append(frame.copy())
                p["post_remaining"] -= 1

        ready = [p for p in self.pending if p["post_remaining"] == 0]
        for p in ready:
            self.pending.remove(p)
            self._dispatch(p)

    def trigger(self, event_type, person_id, side, frame_idx, time_sec):
        snap = {
            "event_type":     event_type,
            "person_id":      int(person_id),
            "side":           side,
            "frame_idx":      int(frame_idx),
            "time_sec":       float(time_sec),
            "pre":            list(self.pre_buffer),
            "post":           [],
            "post_remaining": self.post_frames,
        }
        self.pending.append(snap)

    def _dispatch(self, snap):
        # Prune finished threads to prevent memory leak
        self._threads = [t for t in self._threads if t.is_alive()]
        
        clip_name = (f"{self.video_stem}_{snap['event_type']}_"
                     f"p{snap['person_id']}_f{snap['frame_idx']}.mp4")
        clip_path = os.path.join(CLIPS_DIR, clip_name)
        frames    = snap["pre"] + snap["post"]
        if frames:
            h, w   = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w_out  = cv2.VideoWriter(clip_path, fourcc, self.fps, (w, h))
            for f in frames:
                w_out.write(f)
            w_out.release()
        snap["clip_path"] = clip_path
        t = threading.Thread(target=self._call_vlm, args=(snap,), daemon=True)
        self._threads.append(t)
        t.start()

    def _sample_frames(self, frames):
        if len(frames) <= VLM_SAMPLE_N:
            return frames
        idxs = np.linspace(0, len(frames) - 1, VLM_SAMPLE_N, dtype=int)
        return [frames[i] for i in idxs]

    def _encode_b64(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return base64.b64encode(buf).decode("utf-8")

    def _call_vlm(self, snap):
        if VLM_MOCK_MODE or self.client is None:
            time.sleep(0.3)
            confidence = 0.88 if snap["event_type"] == "DANGER" else 0.62
            result = {
                "verified":    True,
                "is_child":    True,
                "confidence":  confidence,
                "description": (f"[MOCK] Confirmed {snap['event_type'].lower()} — "
                                f"child wrist near stove zone."),
                "mock":        True,
            }
        else:
            try:
                frames  = self._sample_frames(snap["pre"] + snap["post"])
                content = [{"type": "text",
                            "text": self.USER_PROMPT_TEMPLATE.format(
                                event_type=snap["event_type"], n=len(frames))}]
                for f in frames:
                    b64 = self._encode_b64(f)
                    if b64 is None:
                        continue
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

                resp = self.client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": content},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`").lstrip("json").strip()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    result = {"verified": None, "raw_response": raw}
            except Exception as e:
                result = {"verified": None, "error": str(e)}

        result.update({
            "event_type": snap["event_type"],
            "person_id":  snap["person_id"],
            "side":       snap["side"],
            "frame_idx":  snap["frame_idx"],
            "time_sec":   snap["time_sec"],
            "clip_path":  snap["clip_path"],
        })
        with self._results_lock:
            self.results.append(result)

        verdict = result.get("verified")
        desc    = result.get("description") or result.get("raw_response") or result.get("error", "")
        print(f"\n[VLM] {snap['event_type']} f{snap['frame_idx']} "
              f"verified={verdict} conf={result.get('confidence','?')} | {desc}")

    def shutdown(self, timeout=10.0):
        for p in list(self.pending):
            self._dispatch(p)
        self.pending.clear()
        deadline = time.time() + timeout
        for t in self._threads:
            remaining = max(0.1, deadline - time.time())
            if remaining <= 0:
                break
            t.join(timeout=remaining)
        self._threads.clear()

    def get_results(self):
        with self._results_lock:
            return list(self.results)


# =========================================================
# ANNOTATION
# =========================================================
def draw_hud(img, frame_idx, fps, danger_count, touch_count, active_dangers):
    ih, iw = img.shape[:2]
    if ih < 150 or iw < 280:
        return  # Frame too small for HUD overlay
    
    # Sleek dark semi-transparent panel for stats
    x1, y1, x2, y2 = 20, 20, 260, 110
    panel = img[y1:y2, x1:x2].copy()
    img[y1:y2, x1:x2] = cv2.addWeighted(panel, 0.15, np.zeros_like(panel), 0.85, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Use HERSHEY_SIMPLEX for a cleaner, modern look
    cv2.putText(img, f"DANGER : {danger_count}", (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 255), 2)
    cv2.putText(img, f"TOUCH  : {touch_count}", (35, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 255), 2)
    
    # Frame counter below the box
    cv2.putText(img, f"{round(frame_idx/fps,1)}s | F{frame_idx}", (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if active_dangers:
        # Pulsing red danger banner - moved up to avoid media player controls
        alpha = 0.6 + 0.2 * np.sin(frame_idx * 0.4)
        banner_h = 70
        banner_y = ih - banner_h - 40  # Moved up 40px
        
        # Draw the semi-transparent banner FIRST
        banner_crop = img[banner_y:banner_y+banner_h, 0:iw].copy()
        red_bg = np.zeros_like(banner_crop)
        red_bg[:] = (0, 0, 180)
        img[banner_y:banner_y+banner_h, 0:iw] = cv2.addWeighted(banner_crop, 1 - alpha, red_bg, alpha, 0)
        
        # Draw the text ON TOP so it stays pure white
        text = "!! DANGER: CHILD AT STOVE !!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thick = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text,
                    (iw // 2 - tw // 2, banner_y + banner_h // 2 + th // 2),
                    font, scale, (255, 255, 255), thick)

# =========================================================
# ZONE PERSISTENCE (Simplified — No Adult Reference Line)
# =========================================================
def _load_zones() -> dict:
    if not os.path.exists(ZONES_FILE):
        return {}
    with open(ZONES_FILE, "r") as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for k, v in raw.items():
        try:
            poly_data = v["stove"] if isinstance(v, dict) else v
            poly = np.array(poly_data, dtype=np.int32)
            if poly.ndim != 2 or poly.shape[1] != 2:
                print(f"[WARN] Invalid stove polygon shape for '{k}': expected (N, 2), got {poly.shape}")
                continue
            out[k] = {"stove": poly}
        except Exception as e:
            print(f"[WARN] Failed to parse zone '{k}': {e}")
    return out

def _save_zone(video_name: str, poly: np.ndarray):
    zones = {}
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, "r") as f:
            zones = yaml.safe_load(f) or {}
    zones[video_name] = {
        "stove": poly.tolist(),
    }
    with open(ZONES_FILE, "w") as f:
        yaml.safe_dump(zones, f)
    print(f"  Zone saved to {ZONES_FILE}.")

# =========================================================
# INTERACTIVE SETUP — No Adult Shoulder Line Stage
# =========================================================
def get_or_draw_zone(video_name: str, video_path: str, saved_zones: dict) -> dict:
    if video_name in saved_zones:
        z = saved_zones[video_name]
        print(f"  Loaded saved zone for {video_name} ({len(z['stove'])} points).")
        return z

    print(f"  No saved zone for '{video_name}' — draw it now.")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")

    poly_list = []
    done      = [False]

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            poly_list.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if poly_list:
                poly_list.pop()

    WIN = f"Setup Stove Zone: {video_name}"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, callback)

    print("  Draw STOVE polygon")
    print("    L-click: add point | R-click: undo | C: close polygon & finish | Q: abort")

    while not done[0]:
        display = frame.copy()
        h, w    = display.shape[:2]

        # Draw polygon preview
        if poly_list:
            pts = np.array(poly_list, np.int32)
            cv2.polylines(display, [pts], isClosed=done[0], color=(0, 0, 255), thickness=2)
            for pt in poly_list:
                cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)

        cv2.putText(display, f"Draw STOVE polygon  |  Points: {len(poly_list)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "L-click: add | R-click: undo | C: finish | Q: quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Setup aborted.")
        if key == ord('c'):
            if len(poly_list) >= 3:
                done[0] = True
            else:
                print("  Need at least 3 points to close polygon!")

    cv2.destroyAllWindows()
    poly = np.array(poly_list, dtype=np.int32)
    _save_zone(video_name, poly)
    zone_dict = {"stove": poly}
    saved_zones[video_name] = zone_dict
    print(f"  Zone locked: polygon={len(poly_list)}pts\n")
    return zone_dict

# =========================================================
# DEEPSTREAM CONFIG & TEMPLATE BUILDERS
# =========================================================
def generate_default_deepstream_configs():
    """Generates standard template config files for nvinfer and nvtracker if they do not exist."""
    if not os.path.exists(DS_CONFIG_INFER):
        infer_tmpl = f"""[property]
gpu-id=0
net-scale-factor=0.003921569790691523
model-color-format=0
onnx-model=yolov8x-pose.onnx
model-engine-file={DS_MODEL_ENGINE}
labelfile-path=labels.txt
batch-size=1
network-mode=1
num-detected-classes=1
interval=0
gie-unique-id=1
process-mode=1
network-type=100
parse-bbox-func-name=NvDsInferParseCustomYoloPose
custom-lib-path={DS_CUSTOM_PARSER_SO}
"""
        with open(DS_CONFIG_INFER, "w") as f:
            f.write(infer_tmpl)
        # Create a default labels.txt
        if not os.path.exists("labels.txt"):
            with open("labels.txt", "w") as f:
                f.write("person\n")
        print(f"  [DEEPSTREAM] Generated template config: {DS_CONFIG_INFER}")

    if not os.path.exists(DS_CONFIG_TRACKER):
        tracker_tmpl = """[tracker]
enable-batch-process=1
tracker-width=640
tracker-height=384
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF.yml
gpu-id=0
"""
        with open(DS_CONFIG_TRACKER, "w") as f:
            f.write(tracker_tmpl)
        print(f"  [DEEPSTREAM] Generated template config: {DS_CONFIG_TRACKER}")

# =========================================================
# DEEPSTREAM PROBES & METADATA PARSERS
# =========================================================
def extract_yolo_pose_keypoints(obj_meta):
    """
    Extracts 17 keypoint coordinates + confidence from DeepStream custom user metadata.
    Returns (kpts_xy, kpts_conf) where kpts_xy is (17, 2) and kpts_conf is (17,),
    or (None, None) if parsing fails.
    """
    l_user = obj_meta.user_meta_list
    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_USER_META:
            try:
                # The custom parser attaches 17 keypoints * 3 floats (x, y, confidence) = 51 floats total
                raw_data = ctypes.cast(user_meta.user_meta_data, ctypes.POINTER(ctypes.c_float * 51)).contents
                kpts = np.array(raw_data).reshape((17, 3))
                return kpts[:, :2], kpts[:, 2]  # (xy coordinates, confidence scores)
            except Exception as e:
                print(f"[WARN] Failed to parse keypoints from user_meta: {e}")
        l_user = l_user.next
    return None, None

def draw_ds_stove_label(frame, cx, cy, zone_color):
    text_sz = 0.5
    (tw, th), _ = cv2.getTextSize("STOVE", cv2.FONT_HERSHEY_SIMPLEX, text_sz, 2)
    pad = 6
    cv2.rectangle(frame, (int(cx) - tw//2 - pad, int(cy) - th - pad), 
                  (int(cx) + tw//2 + pad, int(cy) + pad), (30, 30, 30), -1)
    cv2.rectangle(frame, (int(cx) - tw//2 - pad, int(cy) - th - pad), 
                  (int(cx) + tw//2 + pad, int(cy) + pad), zone_color, 1)
    cv2.putText(frame, "STOVE", (int(cx) - tw//2, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, text_sz, (255,255,255), 2)

def log_event_ds(event_type, person_id, side, frame_idx, fps, verifier, event_log, u_data):
    time_sec = round(frame_idx / fps, 2)
    event_log.append({
        "event":    event_type,
        "person":   int(person_id),
        "side":     side,
        "frame":    int(frame_idx),
        "time_sec": time_sec,
    })
    if event_type == "DANGER":
        u_data["danger_count"] += 1
    elif event_type == "TOUCH":
        u_data["touch_count"] += 1
    verifier.trigger(event_type, person_id, side, frame_idx, time_sec)
    print(f"\n[DEEPSTREAM ALERT][{event_type}] Person #{person_id} {side} "
          f"@ frame {frame_idx} ({time_sec}s) — VLM verification queued")

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    DeepStream Buffer Probe: Called for each frame batch inside GStreamer.
    Accesses metadata, extracts YOLO pose keypoints, runs child classification and state machine,
    and uses OpenCV to draw the HUD directly on the CUDA hardware frame buffer!
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    # CRITICAL: Make the buffer writable before any OpenCV drawing on the frame surface
    if not gst_buffer.is_writable():
        gst_buffer = gst_buffer.make_writable()

    # Retrieve DeepStream batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # Load parameters passed via u_data
    fps = u_data["fps"]
    stove_poly = u_data["stove_poly"]
    person_states = u_data["person_states"]
    last_seen_frame = u_data["last_seen_frame"]
    stale_archive = u_data["stale_archive"]
    event_log = u_data["event_log"]
    verifier = u_data["verifier"]
    child_stable_frames = u_data["child_stable_frames"]
    
    # Timing constants
    _S = fps / 30.0
    touch_frames = max(5, round(15 * _S))
    danger_frames = max(15, round(90 * _S))
    stale_frames = max(10, round(45 * _S))
    
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        frame_idx = frame_meta.frame_num
        
        # 1. Retrieve the frame image as a NumPy array (mapped GPU CUDA memory)
        # Create a CPU copy for all OpenCV drawing; write back to GPU at the end
        frame = None
        cpu_frame = None
        try:
            frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            cpu_frame = frame.copy()
        except Exception as e:
            print(f"[WARN] Failed to get buffer surface: {e}")
            
        # Prune stale tracks — archive appearance for Re-ID before discarding
        stale = [t for t, lf in last_seen_frame.items() if (frame_idx - lf) > stale_frames]
        for t in stale:
            if REID_ENABLED and t in person_states:
                ps_old = person_states[t]
                if ps_old.get("appearance_hist") is not None:
                    stale_archive[t] = {
                        "histogram":    ps_old["appearance_hist"],
                        "person_state": ps_old,
                        "last_frame":   last_seen_frame[t],
                    }
            person_states.pop(t, None)
            last_seen_frame.pop(t, None)
        
        # Expire old Re-ID archive entries
        if REID_ENABLED and stale_archive:
            _expired = [k for k, v in stale_archive.items()
                        if (frame_idx - v["last_frame"]) / fps > REID_MAX_AGE_SEC]
            for k in _expired:
                del stale_archive[k]
            
        active_dangers = []
        l_obj = frame_meta.obj_meta_list
        
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            
            # Person class (class ID 0 under standard COCO YOLO)
            if obj_meta.class_id == 0:
                track_id = obj_meta.object_id
                last_seen_frame[track_id] = frame_idx

                # Extract keypoints early so spatial Re-ID can use them
                kpts, kpts_conf = extract_yolo_pose_keypoints(obj_meta)
                _ds_rect = obj_meta.rect_params
                _x1 = int(_ds_rect.left)
                _y1 = int(_ds_rect.top)
                _x2 = int(_ds_rect.left + _ds_rect.width)
                _y2 = int(_ds_rect.top + _ds_rect.height)
                
                # Initialize track states if new — with Re-ID matching
                if track_id not in person_states:
                    _matched = False
                    if REID_ENABLED and stale_archive and frame is not None:
                        _new_hist = compute_appearance(frame, (_x1, _y1, _x2, _y2),
                                                       kpts=kpts, confs=kpts_conf)
                        if _new_hist is not None:
                            _best_id, _best_corr = None, -1.0
                            for _old_id, _arch in stale_archive.items():
                                _age = (frame_idx - _arch["last_frame"]) / fps
                                if _age > REID_MAX_AGE_SEC:
                                    continue
                                if _new_hist.shape != _arch["histogram"].shape:
                                    continue
                                _corr = cv2.compareHist(
                                    _new_hist, _arch["histogram"],
                                    cv2.HISTCMP_CORREL)
                                if _corr > _best_corr:
                                    _best_corr = _corr
                                    _best_id   = _old_id
                            if _best_id is not None and _best_corr >= REID_MATCH_THRESH:
                                person_states[track_id] = stale_archive[_best_id]["person_state"]
                                # Reset temporal wrist states across track breaks
                                person_states[track_id]["left_wrist"] = _init_wrist()
                                person_states[track_id]["right_wrist"] = _init_wrist()
                                person_states[track_id]["left_wrist_history"] = deque(maxlen=3)
                                person_states[track_id]["right_wrist_history"] = deque(maxlen=3)
                                del stale_archive[_best_id]
                                _matched = True
                                print(f"\n[DS RE-ID] Track #{track_id} \u2190 #{_best_id} "
                                      f"(corr={_best_corr:.2f})")
                    if not _matched:
                        person_states[track_id] = init_person(child_stable_frames)
                ps = person_states[track_id]
                
                if kpts is not None:
                    # Convert box coordinates
                    rect = obj_meta.rect_params
                    y1, y2 = rect.top, rect.top + rect.height
                    x1, x2 = rect.left, rect.left + rect.width
                    
                    # Update appearance histogram for Re-ID recovery
                    if REID_ENABLED and frame is not None:
                        ps["appearance_hist"] = compute_appearance(
                            frame, (int(x1), int(y1), int(x2), int(y2)),
                            kpts=kpts, confs=kpts_conf)
                    
                    # 3. Perform Child Classification (now with confidence filtering)
                    vote, dbg = classify_child(kpts, kpts_conf, y1, y2, frame_meta.source_frame_height,
                                               stove_poly=stove_poly,
                                               hs_ratios_history=ps["hs_ratios_history"],
                                               return_debug=True)
                    ps["child_vote"].append(vote)
                    if len(ps["child_vote"]) >= 3:
                        ps["is_child"] = sum(ps["child_vote"]) >= len(ps["child_vote"]) / 2
                    
                    # Update visual labels dynamically in DeepStream OSD metadata!
                    label_color = (0, 165, 255) if ps["is_child"] else (0, 200, 0)
                    obj_meta.text_params.display_text = f"{'CHILD' if ps['is_child'] else 'Adult'} #{track_id}"
                    obj_meta.rect_params.border_color.set(label_color[2]/255.0, label_color[1]/255.0, label_color[0]/255.0, 1.0)
                    
                    if ps["is_child"]:
                        # 4. Wrist State Machine
                        for side, wrist_key, kpt_idx in [
                            ("LEFT",  "left_wrist",  LEFT_WRIST),
                            ("RIGHT", "right_wrist", RIGHT_WRIST),
                        ]:
                            pt = kpts[kpt_idx]
                            conf = float(kpts_conf[kpt_idx]) if kpts_conf is not None else 1.0
                            if is_valid(pt, conf):
                                # Smoothing
                                hist_key = f"{wrist_key}_history"
                                ps[hist_key].append((float(pt[0]), float(pt[1])))
                                pts_list = list(ps[hist_key])
                                smoothed_pt = (sum(p[0] for p in pts_list) / len(pts_list),
                                               sum(p[1] for p in pts_list) / len(pts_list))
                                
                                prev_state = ps[wrist_key]["state"]
                                dyn_proximity = max(10, min(int((x2 - x1) * 0.08), 60))
                                
                                ps[wrist_key] = tick_wrist(ps[wrist_key], smoothed_pt, stove_poly,
                                                           touch_frames, danger_frames,
                                                           proximity_px=dyn_proximity)
                                cur_state = ps[wrist_key]["state"]
                                
                                # Trigger alerts & async VLM verification (with cooldown)
                                cooldown = max(30, int(fps * 5))
                                if (cur_state != prev_state and cur_state in ("TOUCH", "DANGER")
                                    and (frame_idx - ps[wrist_key].get("last_trigger_frame", -9999)) > cooldown):
                                    ps[wrist_key]["last_trigger_frame"] = frame_idx
                                    log_event_ds(cur_state, track_id, side, frame_idx, fps, verifier, event_log, u_data)
                                    
                                if cpu_frame is not None:
                                    # Draw wrist points on CPU copy (safe from GPU buffer corruption)
                                    color = STATE_COLOR.get(cur_state, (180, 180, 180))
                                    cv2.circle(cpu_frame, (int(smoothed_pt[0]), int(smoothed_pt[1])), 8, color, -1)
                                    cv2.circle(cpu_frame, (int(smoothed_pt[0]), int(smoothed_pt[1])), 10, (255, 255, 255), 1)
                                    
                        if ps["left_wrist"]["state"] == "DANGER" or ps["right_wrist"]["state"] == "DANGER":
                            active_dangers.append(track_id)
                            
            l_obj = l_obj.next
            
        # 5. Draw global HUD & safety zones on CPU copy, then write back to GPU buffer
        if cpu_frame is not None:
            zone_color = (0, 0, 255) if active_dangers else (40, 140, 200)
            overlay = cpu_frame.copy()
            cv2.fillPoly(overlay, [stove_poly], zone_color)
            cv2.addWeighted(overlay, 0.15, cpu_frame, 0.85, 0, dst=cpu_frame)
            cv2.polylines(cpu_frame, [stove_poly], isClosed=True, color=zone_color, thickness=2)
            cx, cy = stove_poly.mean(axis=0).astype(int)
            draw_ds_stove_label(cpu_frame, cx, cy, zone_color)
            
            draw_hud(cpu_frame, frame_idx, fps, u_data["danger_count"], u_data["touch_count"], active_dangers)
            verifier.push_frame(cpu_frame)
            # Write rendered frame back to GPU-mapped buffer
            if frame is not None:
                np.copyto(frame, cpu_frame)
            
        l_frame = l_frame.next
        
    return Gst.PadProbeReturn.OK

# =========================================================
# DEEPSTREAM PIPELINE LAUNCHER
# =========================================================
def run_deepstream_pipeline(video_path, zone_dict, video_name, live=False):
    """Constructs and launches the hardware-accelerated NVIDIA DeepStream GStreamer pipeline."""
    print(f"\n--- Starting NVIDIA DeepStream Pipeline for {video_name} ---")
    generate_default_deepstream_configs()
    
    # Initialize GStreamer
    Gst.init(None)
    
    # Create GStreamer Pipeline Container
    pipeline = Gst.Pipeline.new("stove-det-pipeline")
    
    # Instantiate elements
    # 1. Source: Ingests local files or RTSP streams dynamically
    source = Gst.ElementFactory.make("nvurisrcbin", "uri-source")
    if not live:
        uri = Path(video_path).resolve().as_uri()
    else:
        uri = video_path
    source.set_property("uri", uri)
    
    # Read source resolution & FPS for streammux and timing
    _src_w, _src_h, fps = 1920, 1080, 30
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        _src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = round(cap.get(cv2.CAP_PROP_FPS))
        if _src_w < 1 or _src_h < 1:
            print(f"  [WARN] Could not read source resolution, defaulting to 1920x1080")
            _src_w, _src_h = 1920, 1080
        if _fps >= 1:
            fps = _fps
        cap.release()
    
    # 2. Batcher Mux: Batches multiple camera streams into a single memory buffer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("width", _src_w)
    streammux.set_property("height", _src_h)
    streammux.set_property("batch-size", 1)
    
    # 3. Model Inference: nvinfer running YOLOv8x-pose via optimized TensorRT Engine
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", DS_CONFIG_INFER)
    
    # 4. Multi-Object Tracker: nvtracker running GPU NvDCF correlation filter
    tracker = Gst.ElementFactory.make("nvtracker", "object-tracker")
    tracker.set_property("config-file-path", DS_CONFIG_TRACKER)
    
    # 5. Core Video Converter
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    
    # 6. On-Screen Display (OSD): draws parsed bounding boxes & labels
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreen-display")
    
    # 7. Sinks & Exporters: Hardware H.264 compression & rolling writer
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
    parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    mux = Gst.ElementFactory.make("qtmux", "mp4-muxer")
    
    stem = os.path.splitext(video_name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_detected.mp4")
    
    if live and LIVE_DISPLAY:
        sink = Gst.ElementFactory.make("nveglglessink", "display-sink")
    else:
        sink = Gst.ElementFactory.make("filesink", "file-sink")
        sink.set_property("location", out_path)
    
    # Add all instantiated elements to the GStreamer Pipeline (with named validation)
    _elements = {
        "nvurisrcbin": source, "nvstreammux": streammux, "nvinfer": pgie,
        "nvtracker": tracker, "nvvideoconvert": nvvidconv, "nvdsosd": nvosd,
        "nvvideoconvert2": nvvidconv2, "nvv4l2h264enc": encoder,
        "h264parse": parser, "qtmux": mux, "filesink": sink,
    }
    for name, elem in _elements.items():
        if elem is None:
            raise RuntimeError(f"[DEEPSTREAM] Failed to create GStreamer element: {name}")
        pipeline.add(elem)
        
    # Link dynamic source: nvurisrcbin uses dynamic pads (pad-added signal callback)
    sinkpad = streammux.get_request_pad("sink_0")
    
    def _on_pad_added(src, new_pad, sinkpad):
        if not sinkpad.is_linked():
            new_pad.link(sinkpad)
    
    source.connect("pad-added", _on_pad_added, sinkpad)
    
    # Link subsequent pipeline elements sequentially
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(encoder)
    encoder.link(parser)
    parser.link(mux)
    mux.link(sink)
    
    # ── Setup Shared Probe Context Data ──
        
    verifier = VLMVerifier(fps=fps, video_stem=stem)
    probe_data = {
        "fps": fps,
        "stove_poly": zone_dict["stove"],
        "person_states": {},
        "last_seen_frame": {},
        "stale_archive": {},
        "event_log": [],
        "verifier": verifier,
        "child_stable_frames": max(5, round(10 * (fps / 30.0))),
        "danger_count": 0,
        "touch_count": 0
    }
    
    # ── Attach Buffer Probe to nvosd's sink pad ──
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, probe_data)
    
    # Create GLib Loop & play GStreamer Pipeline
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    
    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n  [DEEPSTREAM] End-of-stream reached.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n  [DEEPSTREAM ERROR] {err}: {debug}")
            loop.quit()
        return True
        
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    print("  [DEEPSTREAM] Pipeline successfully constructed. Starting GStreamer main loop...")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
        
    # Teardown
    print("  [DEEPSTREAM] Shutting down GStreamer pipeline.")
    pipeline.set_state(Gst.State.NULL)
    verifier.shutdown(timeout=10.0)
    
    # Save output events log
    log_path = os.path.join(OUTPUT_DIR, f"{stem}_events.json")
    with open(log_path, "w") as f:
        json.dump({
            "danger_count": probe_data["danger_count"],
            "touch_count": probe_data["touch_count"],
            "events": probe_data["event_log"],
            "vlm_results": verifier.get_results(),
            "vlm_mock_mode": VLM_MOCK_MODE,
            "live_mode": live,
        }, f, indent=2)
        
    print(f"  [DEEPSTREAM DONE] Outputs successfully written to:")
    print(f"    Video  -> {out_path}")
    print(f"    Events -> {log_path}")

# =========================================================
# PROCESS ONE VIDEO — Orchestrated Entrypoint
# =========================================================
def process_video(video_path, zone_dict, video_name, live=False):
    if DEEPSTREAM_ENABLED:
        if not DEEPSTREAM_AVAILABLE:
            print("\n[WARNING] DeepStream was enabled but Python bindings (gi, pyds) are not available!")
            print("          Falling back to CPU / PyTorch sequential execution pipeline.\n")
            process_video_opencv(video_path, zone_dict, video_name, live)
        else:
            run_deepstream_pipeline(video_path, zone_dict, video_name, live)
    else:
        process_video_opencv(video_path, zone_dict, video_name, live)

def process_video_opencv(video_path, zone_dict, video_name, live=False):
    stove_poly = zone_dict["stove"]
    stem       = os.path.splitext(video_name)[0]

    if live:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"live_{ts}_detected.avi") if LIVE_RECORD else None
        log_path    = os.path.join(OUTPUT_DIR, f"live_{ts}_events.json")
    else:
        output_path = os.path.join(OUTPUT_DIR, f"{stem}_detected.avi")
        log_path    = os.path.join(OUTPUT_DIR, f"{stem}_events.json")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open {video_path}")
        return

    fps    = round(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode_label = "LIVE" if live else video_name
    print(f"\n--- {mode_label} | {width}x{height} @ {fps}fps ---")

    _S                  = fps / 30.0
    touch_frames        = max(5,  round(15 * _S))
    danger_frames       = max(15, round(90 * _S))
    stale_frames        = max(10, round(45 * _S))
    child_stable_frames = max(5,  round(10 * _S))

    writer = None
    if not live or LIVE_RECORD:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[SKIP] Cannot open writer: {output_path}")
            cap.release()
            return

    person_states   = {}
    last_seen_frame = {}
    stale_archive   = {}    # Re-ID: archived appearance + state for disappeared tracks
    event_log       = []
    danger_count    = 0
    touch_count     = 0

    verifier = VLMVerifier(fps=fps, video_stem=stem)
    print(f"  VLM verifier: mock={VLM_MOCK_MODE}  "
          f"pre={VLM_PRE_SEC}s  post={VLM_POST_SEC}s")

    def log_event(event_type, person_id, side, frame_idx):
        nonlocal danger_count, touch_count
        time_sec = round(frame_idx / fps, 2)
        event_log.append({
            "event":    event_type,
            "person":   int(person_id),
            "side":     side,
            "frame":    int(frame_idx),
            "time_sec": time_sec,
        })
        if event_type == "DANGER":
            danger_count += 1
        elif event_type == "TOUCH":
            touch_count += 1
        verifier.trigger(event_type, person_id, side, frame_idx, time_sec)
        print(f"\n[{event_type}] Person #{person_id} {side} "
              f"@ frame {frame_idx} ({time_sec}s) — VLM verification queued")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        print(f"\r  Frame {frame_idx}", end="")

        annotated = frame.copy()

        # Prune stale tracks — archive appearance for Re-ID before discarding
        stale = [t for t, lf in last_seen_frame.items()
                 if (frame_idx - lf) > stale_frames]
        for t in stale:
            if REID_ENABLED and t in person_states:
                ps_old = person_states[t]
                if ps_old.get("appearance_hist") is not None:
                    stale_archive[t] = {
                        "histogram":    ps_old["appearance_hist"],
                        "person_state": ps_old,
                        "last_frame":   last_seen_frame[t],
                    }
            person_states.pop(t, None)
            last_seen_frame.pop(t, None)

        # Expire old Re-ID archive entries
        if REID_ENABLED and stale_archive:
            _expired = [k for k, v in stale_archive.items()
                        if (frame_idx - v["last_frame"]) / fps > REID_MAX_AGE_SEC]
            for k in _expired:
                del stale_archive[k]

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        active_dangers = []

        if results:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                ids       = r.boxes.id.cpu().numpy().astype(int)
                kpts_xy   = r.keypoints.xy.cpu().numpy()
                kpts_conf = r.keypoints.conf.cpu().numpy() \
                            if r.keypoints.conf is not None else None
                conf_iter = kpts_conf if kpts_conf is not None else [None] * len(ids)

                for box, track_id, kpts, confs in zip(boxes, ids, kpts_xy, conf_iter):
                    x1, y1, x2, y2 = map(int, box)
                    last_seen_frame[track_id] = frame_idx

                    if track_id not in person_states:
                        _matched = False
                        if REID_ENABLED and stale_archive:
                            _new_hist = compute_appearance(frame, (x1, y1, x2, y2),
                                                           kpts=kpts, confs=confs)
                            if _new_hist is not None:
                                _best_id, _best_corr = None, -1.0
                                for _old_id, _arch in stale_archive.items():
                                    _age = (frame_idx - _arch["last_frame"]) / fps
                                    if _age > REID_MAX_AGE_SEC:
                                        continue
                                    if _new_hist.shape != _arch["histogram"].shape:
                                        continue
                                    _corr = cv2.compareHist(
                                        _new_hist, _arch["histogram"],
                                        cv2.HISTCMP_CORREL)
                                    if _corr > _best_corr:
                                        _best_corr = _corr
                                        _best_id   = _old_id
                                if _best_id is not None and _best_corr >= REID_MATCH_THRESH:
                                    person_states[track_id] = stale_archive[_best_id]["person_state"]
                                    # Reset temporal wrist states across track breaks
                                    person_states[track_id]["left_wrist"] = _init_wrist()
                                    person_states[track_id]["right_wrist"] = _init_wrist()
                                    person_states[track_id]["left_wrist_history"] = deque(maxlen=3)
                                    person_states[track_id]["right_wrist_history"] = deque(maxlen=3)
                                    del stale_archive[_best_id]
                                    _matched = True
                                    print(f"\n[RE-ID] Track #{track_id} \u2190 #{_best_id} "
                                          f"(corr={_best_corr:.2f})")
                        if not _matched:
                            person_states[track_id] = init_person(child_stable_frames)
                    ps = person_states[track_id]

                    vote, dbg = classify_child(kpts, confs, y1, y2, height,
                                               stove_poly=stove_poly,
                                               hs_ratios_history=ps["hs_ratios_history"],
                                               return_debug=True)
                    ps["child_vote"].append(vote)
                    if len(ps["child_vote"]) >= 3:
                        ps["is_child"] = sum(ps["child_vote"]) >= len(ps["child_vote"]) / 2

                    # Update appearance histogram for Re-ID recovery
                    if REID_ENABLED:
                        ps["appearance_hist"] = compute_appearance(frame, (x1, y1, x2, y2),
                                                                   kpts=kpts, confs=confs)

                    label_color = (0, 165, 255) if ps["is_child"] else (0, 200, 0)
                    label_text  = f"{'CHILD' if ps['is_child'] else 'Adult'} #{track_id}"
                    
                    # Filled background for label for better readability
                    (tw, th), tf = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), label_color, -1)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), label_color, 2)
                    cv2.putText(annotated, label_text, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if DEBUG_CLASSIFIER:
                        dy = y1 + 18
                        if dbg.get("method") == "stove_anchor":
                            cv2.putText(annotated,
                                        f"STOVE_ANCHOR r={dbg['ratio']:.2f} ({dbg['sh_src']})",
                                        (x1 + 4, dy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 200, 0), 1)
                        else:
                            # Falling back to horizontal ratios or missing keys
                            for k in ("hd/sh", "sh/hp", "hd/hp", "bbH"):
                                if k in dbg:
                                    ratio, passed = dbg[k]
                                    txt   = f"{k}={ratio:.2f}{'+' if passed else '-'}"
                                    color = (0, 200, 255) if passed else (120, 120, 120)
                                    cv2.putText(annotated, txt, (x1 + 4, dy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                    dy += 14
                            cv2.putText(annotated,
                                        f"votes {dbg.get('child_votes',0)}/{dbg.get('n_signals',0)}"
                                        f"{' CR' if dbg.get('crouching') else ''}",
                                        (x1 + 4, dy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    if not ps["is_child"]:
                        continue

                    for side, wrist_key, kpt_idx in [
                        ("LEFT",  "left_wrist",  LEFT_WRIST),
                        ("RIGHT", "right_wrist", RIGHT_WRIST),
                    ]:
                        try:
                            pt = kpts[kpt_idx]
                        except IndexError:
                            continue

                        conf = float(confs[kpt_idx]) if confs is not None else 1.0
                        if not is_valid(pt, conf):
                            continue

                        # Smooth wrist coordinate using 3-frame rolling average
                        hist_key = f"{wrist_key}_history"
                        ps[hist_key].append((float(pt[0]), float(pt[1])))
                        
                        pts_list = list(ps[hist_key])
                        smoothed_x = sum(p[0] for p in pts_list) / len(pts_list)
                        smoothed_y = sum(p[1] for p in pts_list) / len(pts_list)
                        smoothed_pt = (smoothed_x, smoothed_y)

                        prev_state    = ps[wrist_key]["state"]
                        # Calculate dynamic proximity padding based on bounding box width (8% of width, constrained between 10 and 60 pixels)
                        bbox_w = abs(x2 - x1)
                        dyn_proximity = max(10, min(int(bbox_w * 0.08), 60))
                        
                        ps[wrist_key] = tick_wrist(ps[wrist_key], smoothed_pt, stove_poly,
                                                   touch_frames, danger_frames,
                                                   proximity_px=dyn_proximity)
                        cur_state     = ps[wrist_key]["state"]

                        cooldown = max(30, int(fps * 5))
                        if (cur_state != prev_state and cur_state in ("TOUCH", "DANGER")
                            and (frame_idx - ps[wrist_key].get("last_trigger_frame", -9999)) > cooldown):
                            ps[wrist_key]["last_trigger_frame"] = frame_idx
                            log_event(cur_state, track_id, side, frame_idx)

                        color = STATE_COLOR.get(cur_state, (180, 180, 180))
                        cv2.circle(annotated, (int(smoothed_pt[0]), int(smoothed_pt[1])), 8,  color,         -1)
                        cv2.circle(annotated, (int(smoothed_pt[0]), int(smoothed_pt[1])), 10, (255, 255, 255), 1)

                    if (ps["left_wrist"]["state"] == "DANGER" or
                            ps["right_wrist"]["state"] == "DANGER"):
                        active_dangers.append(track_id)

        # Draw stove zone with subtle semi-transparent fill
        zone_color = (0, 0, 255) if active_dangers else (40, 140, 200)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [stove_poly], zone_color)
        # Reduced alpha for the fill so it doesn't wash out the image
        annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)
        
        # Draw clean border
        cv2.polylines(annotated, [stove_poly], isClosed=True,
                      color=zone_color, thickness=2)
        cx, cy = stove_poly.mean(axis=0).astype(int)
        
        # Sleek Stove label with dark background
        text_sz = 0.5
        (tw, th), _ = cv2.getTextSize("STOVE", cv2.FONT_HERSHEY_SIMPLEX, text_sz, 2)
        pad = 6
        cv2.rectangle(annotated, (int(cx) - tw//2 - pad, int(cy) - th - pad), 
                      (int(cx) + tw//2 + pad, int(cy) + pad), (30, 30, 30), -1)
        # Add a thin colored border to the label box
        cv2.rectangle(annotated, (int(cx) - tw//2 - pad, int(cy) - th - pad), 
                      (int(cx) + tw//2 + pad, int(cy) + pad), zone_color, 1)
        cv2.putText(annotated, "STOVE", (int(cx) - tw//2, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_sz, (255,255,255), 2)

        draw_hud(annotated, frame_idx, fps, danger_count, touch_count, active_dangers)
        if writer is not None:
            writer.write(annotated)
        verifier.push_frame(annotated)

        # Live mode: display window + quit check
        if live and LIVE_DISPLAY:
            cv2.imshow("Stove Detection - LIVE", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  [LIVE] Quit requested.")
                break

    cap.release()
    if writer is not None:
        writer.release()
    if live:
        cv2.destroyAllWindows()

    print(f"\n  Waiting for VLM verifications to finish...")
    verifier.shutdown(timeout=15.0)
    vlm_results = verifier.get_results()

    with open(log_path, "w") as f:
        json.dump({
            "danger_count":   danger_count,
            "touch_count":    touch_count,
            "events":         event_log,
            "vlm_results":    vlm_results,
            "vlm_mock_mode":  VLM_MOCK_MODE,
            "live_mode":      live,
        }, f, indent=2)

    verified_count = sum(1 for r in vlm_results if r.get("verified") is True)
    print(f"\n  Done. DANGER={danger_count}  TOUCH={touch_count}  "
          f"VLM_VERIFIED={verified_count}/{len(vlm_results)}")
    if output_path:
        print(f"  Video  -> {output_path}")
    print(f"  Events -> {log_path}")
    print(f"  Clips  -> {CLIPS_DIR}/")

# =========================================================
# MAIN
# =========================================================
saved_zones = _load_zones()

if LIVE_MODE:
    # ── Live stream mode ──
    try:
        src = int(LIVE_SOURCE)
    except ValueError:
        src = LIVE_SOURCE
    stream_name = f"webcam_{src}" if isinstance(src, int) else "live_stream"
    print(f"\n[LIVE] Opening source: {LIVE_SOURCE}")
    zone = get_or_draw_zone(stream_name, src, saved_zones)
    process_video(src, zone, stream_name, live=True)
    print(f"\n{'='*50}")
    print(f"  Live session ended.")
    print(f"  Outputs in: {OUTPUT_DIR}/")
    print(f"{'='*50}")
else:
    # ── Batch file mode (default) ──
    for i, video_name in enumerate(VIDEO_FILES, start=1):
        video_path = os.path.join(INPUT_DIR, video_name)
        print(f"\n[{i}/{len(VIDEO_FILES)}] {video_name}")
        zone = get_or_draw_zone(video_name, video_path, saved_zones)
        process_video(video_path, zone, video_name)

    print(f"\n{'='*50}")
    print(f"  All {len(VIDEO_FILES)} video(s) processed using finalized dynamic child detection.")
    print(f"  Outputs in: {OUTPUT_DIR}/")
    print(f"{'='*50}")
