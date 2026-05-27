import os
import cv2
import json
import yaml
import time
import base64
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque

# =========================================================
# PATHS
# =========================================================
MODEL_PATH  = "yolo26l-pose.pt"
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
print(f"\nLoading {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

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

# =========================================================
# RE-ID — Appearance Histogram
# =========================================================
def compute_appearance(frame, bbox):
    """Compute a normalized HSV color histogram for a bounding box region.
    Used for Re-ID matching when ByteTrack loses a track after occlusion."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2],
                        None,
                        [REID_HIST_BINS, REID_HIST_BINS, REID_HIST_BINS],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

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

def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

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
        l_kn = _kpt(kpts, confs, LEFT_KNEE)
        r_kn = _kpt(kpts, confs, RIGHT_KNEE)
        l_hp = _kpt(kpts, confs, LEFT_HIP)
        r_hp = _kpt(kpts, confs, RIGHT_HIP)

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
        elif l_hp and r_hp:
            y_feet = (l_hp[1] + r_hp[1]) / 2.0 + 80.0; feet_src = "hips_offset"
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
                # Decision threshold: 1.15 (conservatively separates kids at <1.15 from adults)
                is_child = ratio < 1.15

                if return_debug:
                    return is_child, {
                        "method":      "stove_anchor",
                        "sh_src":      sh_src,
                        "feet_src":    feet_src,
                        "ratio":       ratio,
                        "y_feet":      y_feet,
                        "y_shoulder":  y_shoulder,
                        "y_stove_top": y_stove_top,
                        "result":      is_child,
                    }
                return is_child

    # ─── Fallback Crouch-Robust Classifier (Horizontal Ratios + Temporal Filtering) ───
    hip_w = abs(r_hip[0] - l_hip[0]) if (l_hip and r_hip) else None

    crouching = False
    if sh_mid and hip_mid and ankle_mid:
        torso_y = abs(hip_mid[1] - sh_mid[1])
        leg_y   = abs(ankle_mid[1] - hip_mid[1])
        if torso_y > 5:
            crouching = leg_y < torso_y * 0.85

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
    return {"frames_in_zone": 0, "state": "SAFE"}

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
        threading.Thread(target=self._call_vlm, args=(snap,), daemon=True).start()

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
        while time.time() < deadline:
            alive = [t for t in threading.enumerate()
                     if t.daemon and t.name.startswith("Thread") and t.is_alive()]
            if not alive:
                break
            time.sleep(0.2)

    def get_results(self):
        with self._results_lock:
            return list(self.results)


# =========================================================
# ANNOTATION
# =========================================================
def draw_hud(img, frame_idx, fps, danger_count, touch_count, active_dangers):
    ih, iw = img.shape[:2]
    
    # Sleek dark semi-transparent panel for stats
    x1, y1, x2, y2 = 15, 15, 300, 115
    panel = img[y1:y2, x1:x2].copy()
    img[y1:y2, x1:x2] = cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    cv2.putText(img, f"DANGER  : {danger_count}", (30, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 255), 2)
    cv2.putText(img, f"TOUCH   : {touch_count}", (30, 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 170, 255), 2)
    cv2.putText(img, f"Frame {frame_idx} | {round(frame_idx/fps,1)}s", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if active_dangers:
        # Pulsing red danger banner at bottom
        alpha = 0.7 + 0.3 * np.sin(frame_idx * 0.4)
        banner_h = 60
        banner_y = ih - banner_h
        banner_crop = img[banner_y:ih, 0:iw].copy()
        
        red_bg = np.zeros_like(banner_crop)
        red_bg[:] = (0, 0, 180)
        img[banner_y:ih, 0:iw] = cv2.addWeighted(banner_crop, 1 - alpha, red_bg, alpha, 0)
        
        text = "!! DANGER: CHILD AT STOVE !!"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thick = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text,
                    (iw // 2 - tw // 2, ih - banner_h // 2 + th // 2),
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
        if isinstance(v, dict):
            out[k] = {
                "stove": np.array(v["stove"], dtype=np.int32),
            }
        else:
            # Backward compatibility with bare list of points
            out[k] = {"stove": np.array(v, dtype=np.int32)}
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
# PROCESS ONE VIDEO
# =========================================================
def process_video(video_path, zone_dict, video_name, live=False):
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
                            _new_hist = compute_appearance(frame, (x1, y1, x2, y2))
                            if _new_hist is not None:
                                _best_id, _best_corr = None, -1.0
                                for _old_id, _arch in stale_archive.items():
                                    _age = (frame_idx - _arch["last_frame"]) / fps
                                    if _age > REID_MAX_AGE_SEC:
                                        continue
                                    _corr = cv2.compareHist(
                                        _new_hist, _arch["histogram"],
                                        cv2.HISTCMP_CORREL)
                                    if _corr > _best_corr:
                                        _best_corr = _corr
                                        _best_id   = _old_id
                                if _best_id is not None and _best_corr >= REID_MATCH_THRESH:
                                    person_states[track_id] = stale_archive[_best_id]["person_state"]
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
                        ps["appearance_hist"] = compute_appearance(frame, (x1, y1, x2, y2))

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

                        if cur_state != prev_state and cur_state in ("TOUCH", "DANGER"):
                            log_event(cur_state, track_id, side, frame_idx)

                        color = STATE_COLOR.get(cur_state, (180, 180, 180))
                        cv2.circle(annotated, (int(smoothed_pt[0]), int(smoothed_pt[1])), 8,  color,         -1)
                        cv2.circle(annotated, (int(smoothed_pt[0]), int(smoothed_pt[1])), 10, (255, 255, 255), 1)

                    if (ps["left_wrist"]["state"] == "DANGER" or
                            ps["right_wrist"]["state"] == "DANGER"):
                        active_dangers.append(track_id)

        # Draw stove zone with semi-transparent fill
        zone_color = (0, 0, 255) if active_dangers else (40, 140, 200)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [stove_poly], zone_color)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
        cv2.polylines(annotated, [stove_poly], isClosed=True,
                      color=zone_color, thickness=2)
        cx, cy = stove_poly.mean(axis=0).astype(int)
        
        # Stove label with dark background
        (tw, th), _ = cv2.getTextSize("STOVE", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (int(cx) - tw//2 - 5, int(cy) - th - 5), 
                      (int(cx) + tw//2 + 5, int(cy) + 5), (0, 0, 0), -1)
        cv2.putText(annotated, "STOVE", (int(cx) - tw//2, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

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
