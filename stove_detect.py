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

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR,  exist_ok=True)

# Standard ultralytics model names auto-download on YOLO() init.
# Only fail here if a custom path was given that doesn't exist.
if os.sep in MODEL_PATH and not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

_supported = (".mp4", ".avi", ".mov", ".mkv")
VIDEO_FILES = sorted(
    f for f in os.listdir(INPUT_DIR) if f.lower().endswith(_supported)
)
if not VIDEO_FILES:
    raise FileNotFoundError(f"No videos found in '{INPUT_DIR}/'. Add videos and rerun.")

print(f"Found {len(VIDEO_FILES)} video(s): {VIDEO_FILES}")

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

def is_near_stove(pt, poly):
    dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)
    return dist >= -PROXIMITY_PX

# =========================================================
# CHILD DETECTION — uses limb proportions (crouch-invariant)
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
                   adult_line_y=None, return_debug=False):
    """
    PRIMARY: Environmental anchor — if `adult_line_y` is set, compare
    the person's upper-body keypoint Y against it. Above line (smaller Y)
    = adult. Below line = child. Robust to crouching, perspective, and
    leg occlusion (kitchen counter blocking lower body).

    FALLBACK: Anthropometric horizontal ratios (used when line is None
    or all upper-body keypoints are missing).
    """
    # ─── Environmental anchor (overrides ratios when available) ───
    if adult_line_y is not None:
        nose = _kpt(kpts, confs, NOSE)
        l_sh = _kpt(kpts, confs, LEFT_SHOULDER)
        r_sh = _kpt(kpts, confs, RIGHT_SHOULDER)

        ref_y = None
        ref_src = ""
        if l_sh and r_sh:
            ref_y = (l_sh[1] + r_sh[1]) / 2.0; ref_src = "shoulder_mid"
        elif nose:
            ref_y = nose[1]; ref_src = "nose"
        elif l_sh:
            ref_y = l_sh[1]; ref_src = "L_shoulder"
        elif r_sh:
            ref_y = r_sh[1]; ref_src = "R_shoulder"

        if ref_y is not None:
            # Image Y grows downward — shoulder BELOW line (larger Y) = child
            is_child = ref_y > adult_line_y
            if return_debug:
                return is_child, {
                    "method":   "anchor",
                    "ref_src":  ref_src,
                    "ref_y":    ref_y,
                    "line_y":   adult_line_y,
                    "result":   is_child,
                }
            return is_child
        # Fall through to ratio classifier if no upper keypoints visible


    """
    Crouch-robust classifier — uses HORIZONTAL measurements only.

    Why: vertical body length foreshortens in 2D when crouching,
    making adults look like big-headed children. Horizontal distances
    (head width, shoulder width, hip width) stay roughly constant
    regardless of body posture.

    Adult vs child reference ratios:
      - head_width / shoulder_width : kids ~0.55-0.75, adults ~0.30-0.45
      - shoulder_width / hip_width  : kids ~1.0-1.15,  adults ~1.25-1.45
      - bbox_height frame ratio     : only used when standing (fallback)
    """
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

    # Head width: prefer ear-to-ear (most accurate),
    # fallback to eye-to-eye scaled up (eyes sit ~70% inside ears).
    head_w = None
    if l_ear is not None and r_ear is not None:
        head_w = abs(r_ear[0] - l_ear[0])
    elif l_eye is not None and r_eye is not None:
        head_w = abs(r_eye[0] - l_eye[0]) * 1.6

    shoulder_w = abs(r_sh[0]  - l_sh[0])  if (l_sh and r_sh)   else None
    hip_w      = abs(r_hip[0] - l_hip[0]) if (l_hip and r_hip) else None

    # Crouch detection (used only to disable bbox-height vote)
    crouching = False
    if sh_mid and hip_mid and ankle_mid:
        torso_y = abs(hip_mid[1] - sh_mid[1])
        leg_y   = abs(ankle_mid[1] - hip_mid[1])
        if torso_y > 5:
            crouching = leg_y < torso_y * 0.85

    debug = {}
    scores = []

    if head_w is not None and shoulder_w and shoulder_w > 5:
        r = head_w / shoulder_w
        v = r > 0.45
        scores.append(v); debug["hd/sh"] = (r, v)

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
        "child_vote":  deque(maxlen=child_stable_frames),
        "is_child":    False,
        "left_wrist":  _init_wrist(),
        "right_wrist": _init_wrist(),
    }

def _init_wrist():
    return {"frames_in_zone": 0, "state": "SAFE"}

def tick_wrist(ws, pt, poly, touch_frames, danger_frames):
    if is_near_stove(pt, poly):
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
        """Call every frame with the raw (or annotated) image."""
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
        """Call when a TOUCH or DANGER event fires."""
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
            # Mock result varies by event type for realistic demo output
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
                # Strip markdown fences if present
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
        """Flush pending with whatever post frames are available, then wait."""
        for p in list(self.pending):
            self._dispatch(p)
        self.pending.clear()
        # Best-effort wait for background API threads
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
    x1, y1, x2, y2 = 10, 10, 380, 135
    panel = img[y1:y2, x1:x2].copy()
    img[y1:y2, x1:x2] = cv2.addWeighted(np.zeros_like(panel), 0.7, panel, 0.3, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    cv2.putText(img, f"DANGER EVENTS : {danger_count}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, f"TOUCH EVENTS  : {touch_count}", (20, 90),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(img, f"Frame {frame_idx}  {round(frame_idx/fps,1)}s", (20, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

    if active_dangers:
        banner_y = ih - 55
        cv2.rectangle(img, (0, banner_y), (iw, ih), (0, 0, 160), -1)
        cv2.putText(img, "!! DANGER: CHILD AT STOVE !!",
                    (iw // 2 - 230, ih - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

# =========================================================
# ZONE PERSISTENCE
# =========================================================
def _load_zones() -> dict:
    """
    Returns {video_name: {"stove": np.ndarray(Nx2), "adult_line_y": int|None}}.
    Supports legacy format (bare list of points) for backward compatibility.
    """
    if not os.path.exists(ZONES_FILE):
        return {}
    with open(ZONES_FILE, "r") as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = {
                "stove":        np.array(v["stove"], dtype=np.int32),
                "adult_line_y": v.get("adult_line_y"),
            }
        else:
            # Legacy: list of points only, no adult line set
            out[k] = {"stove": np.array(v, dtype=np.int32), "adult_line_y": None}
    return out

def _save_zone(video_name: str, poly: np.ndarray, adult_line_y):
    zones = {}
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, "r") as f:
            zones = yaml.safe_load(f) or {}
    zones[video_name] = {
        "stove":        poly.tolist(),
        "adult_line_y": int(adult_line_y) if adult_line_y is not None else None,
    }
    with open(ZONES_FILE, "w") as f:
        yaml.safe_dump(zones, f)
    print(f"  Zone saved to {ZONES_FILE}.")

# =========================================================
# INTERACTIVE SETUP — per video, saves stove polygon + adult line
# =========================================================
def _draw_adult_line_only(video_name: str, video_path: str,
                          stove_poly: np.ndarray, saved_zones: dict) -> dict:
    """Prompt user only for the adult shoulder line — polygon already exists."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")

    adult_line_y = [None]
    done         = [False]

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            adult_line_y[0] = y

    WIN = f"Set Adult Line: {video_name}"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, callback)
    print("  Click at ADULT shoulder height | C: confirm | S: skip | Q: abort")

    while not done[0]:
        display = frame.copy()
        h, w    = display.shape[:2]
        cv2.polylines(display, [stove_poly], isClosed=True,
                      color=(0, 0, 255), thickness=2)
        if adult_line_y[0] is not None:
            ly = int(adult_line_y[0])
            cv2.line(display, (0, ly), (w, ly), (255, 200, 0), 2)
            cv2.putText(display, f"ADULT LINE  y={ly}",
                        (10, max(ly - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        cv2.putText(display, "Click at ADULT shoulder height",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "L: place | C: confirm | S: skip | Q: quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Setup aborted.")
        if key == ord('c'):
            if adult_line_y[0] is not None:
                done[0] = True
            else:
                print("  Click to place line first, or press S to skip.")
        if key == ord('s'):
            adult_line_y[0] = None
            done[0] = True

    cv2.destroyAllWindows()
    _save_zone(video_name, stove_poly, adult_line_y[0])
    z = {"stove": stove_poly, "adult_line_y": adult_line_y[0]}
    saved_zones[video_name] = z
    print(f"  Adult line set: y={adult_line_y[0]}\n")
    return z

def get_or_draw_zone(video_name: str, video_path: str, saved_zones: dict) -> dict:
    if video_name in saved_zones:
        z = saved_zones[video_name]
        if z.get("adult_line_y") is not None:
            print(f"  Loaded saved zone for {video_name} "
                  f"({len(z['stove'])} points, adult_line_y={z['adult_line_y']}).")
            return z
        # Polygon exists but adult line missing — prompt for line only
        print(f"  Zone loaded for {video_name} but adult line missing — set it now.")
        return _draw_adult_line_only(video_name, video_path, z["stove"], saved_zones)

    print(f"  No saved zone for '{video_name}' — draw it now.")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame of {video_path}")

    poly_list      = []
    adult_line_y   = [None]   # mutable container
    stage          = ["POLY"] # "POLY" -> "LINE" -> done
    done           = [False]

    def callback(event, x, y, flags, param):
        if stage[0] == "POLY":
            if event == cv2.EVENT_LBUTTONDOWN:
                poly_list.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if poly_list:
                    poly_list.pop()
        elif stage[0] == "LINE":
            if event == cv2.EVENT_LBUTTONDOWN:
                adult_line_y[0] = y
            elif event == cv2.EVENT_MOUSEMOVE and adult_line_y[0] is None:
                # Hover preview before click
                adult_line_y[0] = y if flags == 0 else adult_line_y[0]

    WIN = f"Setup: {video_name}"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, callback)

    print("  STAGE 1 — Stove polygon")
    print("    L-click: add point | R-click: undo | C: close polygon | Q: abort")
    print("  STAGE 2 — Adult shoulder line")
    print("    L-click anywhere at adult-shoulder height | C: confirm | S: skip line")

    while not done[0]:
        display = frame.copy()
        h, w    = display.shape[:2]

        # Always draw polygon (preview during stage 1, locked after)
        if poly_list:
            pts = np.array(poly_list, np.int32)
            cv2.polylines(display, [pts],
                          isClosed=(stage[0] == "LINE"),
                          color=(0, 0, 255), thickness=2)
            for pt in poly_list:
                cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)

        if stage[0] == "POLY":
            cv2.putText(display, f"[1/2] Draw STOVE polygon  |  Points: {len(poly_list)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "L: add | R: undo | C: close | Q: quit",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        elif stage[0] == "LINE":
            # Draw current line preview
            if adult_line_y[0] is not None:
                ly = int(adult_line_y[0])
                cv2.line(display, (0, ly), (w, ly), (255, 200, 0), 2)
                cv2.putText(display, f"ADULT LINE  y={ly}",
                            (10, max(ly - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            cv2.putText(display, "[2/2] Click at ADULT shoulder height",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "L: place line | C: confirm | S: skip | Q: quit",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Setup aborted.")
        if key == ord('c'):
            if stage[0] == "POLY":
                if len(poly_list) >= 3:
                    stage[0] = "LINE"
                else:
                    print("  Need at least 3 points to close polygon!")
            elif stage[0] == "LINE":
                if adult_line_y[0] is not None:
                    done[0] = True
                else:
                    print("  Click to place adult line first, or press S to skip.")
        if key == ord('s') and stage[0] == "LINE":
            adult_line_y[0] = None
            done[0] = True

    cv2.destroyAllWindows()
    poly = np.array(poly_list, dtype=np.int32)
    _save_zone(video_name, poly, adult_line_y[0])
    zone_dict = {"stove": poly, "adult_line_y": adult_line_y[0]}
    saved_zones[video_name] = zone_dict
    print(f"  Zone locked: polygon={len(poly_list)}pts  adult_line_y={adult_line_y[0]}\n")
    return zone_dict

# =========================================================
# PROCESS ONE VIDEO
# =========================================================
def process_video(video_path, zone_dict, video_name):
    stove_poly   = zone_dict["stove"]
    adult_line_y = zone_dict.get("adult_line_y")
    stem         = os.path.splitext(video_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_detected.avi")
    log_path    = os.path.join(OUTPUT_DIR, f"{stem}_events.json")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open {video_path}")
        return

    fps    = round(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n--- {video_name} | {width}x{height} @ {fps}fps ---")

    _S                  = fps / 30.0
    touch_frames        = max(5,  round(15 * _S))
    danger_frames       = max(15, round(90 * _S))
    stale_frames        = max(10, round(45 * _S))
    child_stable_frames = max(5,  round(10 * _S))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[SKIP] Cannot open writer: {output_path}")
        cap.release()
        return

    person_states   = {}
    last_seen_frame = {}
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
        # Queue a VLM clip verification for this event.
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

        # Prune stale tracks
        stale = [t for t, lf in last_seen_frame.items()
                 if (frame_idx - lf) > stale_frames]
        for t in stale:
            person_states.pop(t, None)
            last_seen_frame.pop(t, None)

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
                        person_states[track_id] = init_person(child_stable_frames)
                    ps = person_states[track_id]

                    vote, dbg = classify_child(kpts, confs, y1, y2, height,
                                               adult_line_y=adult_line_y,
                                               return_debug=True)
                    ps["child_vote"].append(vote)
                    if len(ps["child_vote"]) >= 3:
                        ps["is_child"] = sum(ps["child_vote"]) >= len(ps["child_vote"]) / 2

                    label_color = (0, 165, 255) if ps["is_child"] else (0, 200, 0)
                    label_text  = f"{'CHILD' if ps['is_child'] else 'Adult'} #{track_id}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), label_color, 2)
                    cv2.putText(annotated, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

                    if DEBUG_CLASSIFIER:
                        dy = y1 + 18
                        if dbg.get("method") == "anchor":
                            cv2.putText(annotated,
                                        f"ANCHOR {dbg['ref_src']} y={int(dbg['ref_y'])}"
                                        f" vs line={dbg['line_y']}",
                                        (x1 + 4, dy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 200, 0), 1)
                        else:
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
                            # Skip frame — keep in-progress timer so single
                            # missed pose detection doesn't reset TOUCH/DANGER.
                            continue

                        prev_state    = ps[wrist_key]["state"]
                        ps[wrist_key] = tick_wrist(ps[wrist_key], pt, stove_poly,
                                                   touch_frames, danger_frames)
                        cur_state     = ps[wrist_key]["state"]

                        if cur_state != prev_state and cur_state in ("TOUCH", "DANGER"):
                            log_event(cur_state, track_id, side, frame_idx)

                        color = STATE_COLOR.get(cur_state, (180, 180, 180))
                        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8,  color,         -1)
                        cv2.circle(annotated, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), 1)

                    if (ps["left_wrist"]["state"] == "DANGER" or
                            ps["right_wrist"]["state"] == "DANGER"):
                        active_dangers.append(track_id)

        # Adult shoulder reference line (if defined for this video)
        if adult_line_y is not None:
            cv2.line(annotated, (0, adult_line_y),
                     (annotated.shape[1], adult_line_y),
                     (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, "ADULT LINE", (10, max(adult_line_y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Draw stove zone last so it sits on top of person boxes,
        # colored by current frame's active danger state.
        zone_color = (0, 0, 255) if active_dangers else (40, 140, 200)
        cv2.polylines(annotated, [stove_poly], isClosed=True,
                      color=zone_color, thickness=2)
        cx, cy = stove_poly.mean(axis=0).astype(int)
        cv2.putText(annotated, "STOVE", (int(cx) - 25, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        draw_hud(annotated, frame_idx, fps, danger_count, touch_count, active_dangers)
        writer.write(annotated)

        # Feed annotated frame into VLM rolling buffer + any pending post-windows
        verifier.push_frame(annotated)

    cap.release()
    writer.release()

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
        }, f, indent=2)

    verified_count = sum(1 for r in vlm_results if r.get("verified") is True)
    print(f"\n  Done. DANGER={danger_count}  TOUCH={touch_count}  "
          f"VLM_VERIFIED={verified_count}/{len(vlm_results)}")
    print(f"  Video  -> {output_path}")
    print(f"  Events -> {log_path}")
    print(f"  Clips  -> {CLIPS_DIR}/")

# =========================================================
# MAIN
# =========================================================
saved_zones = _load_zones()

for i, video_name in enumerate(VIDEO_FILES, start=1):
    video_path = os.path.join(INPUT_DIR, video_name)
    print(f"\n[{i}/{len(VIDEO_FILES)}] {video_name}")
    zone = get_or_draw_zone(video_name, video_path, saved_zones)
    process_video(video_path, zone, video_name)

print(f"\n{'='*50}")
print(f"  All {len(VIDEO_FILES)} video(s) processed.")
print(f"  Outputs in: {OUTPUT_DIR}/")
print(f"{'='*50}")
