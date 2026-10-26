import csv
import io
import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    redirect,
    url_for,
    send_from_directory,
    request
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# This application is multi-class ready: every class present in the custom
# waste model is treated as a waste category. A separate COCO detector is
# used only as a bottle fallback. New categories require a model trained
# with those categories; code alone cannot create missing classes.


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

DATABASE_FOLDER = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE = os.path.join(
    DATABASE_FOLDER,
    "waste_detection.db"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

existing_evidence_files = [
    name
    for name in os.listdir(RESULT_FOLDER)
    if name.lower().endswith((".jpg", ".jpeg", ".png"))
]
print("Evidence folder:", RESULT_FOLDER)
print("Existing evidence files:", len(existing_evidence_files))


# ============================================================
# MODELS
# ============================================================

WASTE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "waste_detection_fixed",
    "weights",
    "best.pt"
)

POSE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n-pose.pt"
)

BOTTLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)


# ============================================================
# SETTINGS
# ============================================================

WASTE_CONFIDENCE = 0.15
PERSON_CONFIDENCE = 0.50

CAMERA_ID = "CAM-01"
CAMERA_LOCATION = "Paralakhemundi, Odisha"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
CAMERA_BUFFER_SIZE = 1
CAMERA_READ_FAILURE_LIMIT = 30
CAMERA_RECONNECT_DELAY = 0.5

# Camera source:
# "PC"    -> built-in/USB webcam (index 0)
# "PHONE" -> IP/MJPEG stream from your phone camera
CAMERA_MODE = "PHONE"
PHONE_CAMERA_URL = os.environ.get("PHONE_CAMERA_URL", "http://192.168.137.204:8080/video")
print("PHONE CAMERA STREAM:", PHONE_CAMERA_URL)

INFERENCE_EVERY_N_FRAMES = 2
INFERENCE_IMAGE_SIZE = 512
JPEG_QUALITY = 65

NORMAL_SAVE_INTERVAL = 15

# Holding & release confirmation:
# Tuned for fast, accurate response when someone drops or tosses waste:
REQUIRED_HOLDING_FRAMES = 2
REQUIRED_RELEASE_FRAMES = 2

HAND_WASTE_DISTANCE = 90    # Hand/wrist within 90px of waste item
RELEASE_DISTANCE = 85       # Must separate beyond 85px to qualify as release
MIN_RELEASE_MOTION = 12     # Must move at least 12px
FAST_RELEASE_MOTION = 25
MIN_DOWNWARD_DROP_MOTION = 10 # Downward (towards floor) movement >= 10px
BOTTLE_CONFIDENCE = 0.15

DUMPING_EVENT_COOLDOWN = 10
ALERT_VISIBLE_SECONDS = 8   # Alert resets after 8 seconds (not locked for 30s)
LEGAL_ALERT_VISIBLE_SECONDS = 8
NO_PERSON_RELEASE_GRACE_FRAMES = 6

# Waste association / out-of-frame release tuning.
MAX_HOLDING_ASSOCIATION_DISTANCE = 200
MISSING_WASTE_RELEASE_FRAMES = 4
RELEASE_DEBUG_PRINT_EVERY = 1

# Disable non-waste generic motion pixel blobs from triggering illegal dumping!
ENABLE_MOTION_FALLBACK_DUMPING = False

# ============================================================
# DUSTBIN ZONE (LEGAL DISPOSAL AREA)
# ============================================================
# Waste deposited INSIDE the dustbin is treated as PROPER DISPOSAL (LEGAL).
# Waste thrown/dropped OUTSIDE the dustbin triggers ILLEGAL DUMPING!
DUSTBIN_ZONE_ENABLED = True
DUSTBIN_ZONE = None  # None by default: dynamically locks onto the physical bucket only!
AUTO_DETECT_DUSTBIN = True
ILLEGAL_DUMPING_SAVE_COOLDOWN = 10  # Seconds between auto-saving evidence of illegal dumping
last_illegal_dumping_save_time = 0
DUSTBIN_ZONE_WARMUP_FRAMES = 30
DUSTBIN_ZONE_CONFIRM_FRAMES = 5
DUSTBIN_ZONE_ENTRY_MOTION = 25
DUSTBIN_ZONE_MATCH_DISTANCE = 90
DUSTBIN_ZONE_RELEASE_COOLDOWN = 20
DUSTBIN_ZONE_CANDIDATE_MATCH = 110
DUSTBIN_ZONE_CANDIDATE_MAX_GAP = 4
PERSONLESS_REQUIRED_EMPTY_FRAMES = 8

# Strict person-independent dumping confirmation. A stationary/newly
# detected object must show real downward trajectory motion; detector jitter alone
# must not trigger an incident.
PERSONLESS_MIN_CUMULATIVE_MOTION = 60.0
PERSONLESS_MIN_FRAME_MOTION = 22.0
PERSONLESS_CONFIRM_FRAMES = 4
PERSONLESS_MAX_GAP = 2
PERSONLESS_EVENT_COOLDOWN = 30.0
PERSONLESS_MIN_CONFIDENCE = 0.30
PERSONLESS_REQUIRE_DOWNWARD_OR_LARGE_MOTION = True

# Generic motion/trajectory fallback: tracks moving foreground objects even when
# the custom WASTE model does not recognize the object's category. Camera should
# be mounted securely; a moving camera can invalidate background-motion cues.
MOTION_FRAME_WIDTH = CAMERA_WIDTH // 2
MOTION_FRAME_HEIGHT = CAMERA_HEIGHT // 2
MOTION_WARMUP_FRAMES = 45
MOTION_MIN_CONTOUR_AREA = 80
MOTION_MAX_CONTOUR_AREA = 4500
MOTION_MAX_FOREGROUND_RATIO = 0.12
MOTION_TRACK_MAX_STEP = 110.0
MOTION_MAX_GAP = 1
MOTION_MIN_TRACK_FRAMES = 6
MOTION_MIN_PATH_PIXELS = 135.0
MOTION_MIN_DISPLACEMENT_PIXELS = 75.0
MOTION_FAST_PATH_PIXELS = 240.0
MOTION_FAST_DISPLACEMENT_PIXELS = 140.0
MOTION_EVENT_COOLDOWN = 30.0
MOTION_PERSON_PAD_X_RATIO = 0.12
MOTION_PERSON_PAD_Y_RATIO = 0.05

motion_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=240,
    varThreshold=32,
    detectShadows=False
)
motion_warmup_counter = 0
motion_candidate_center = None
motion_candidate_start_center = None
motion_candidate_last_center = None
motion_candidate_box = None
motion_candidate_streak = 0
motion_candidate_gap = 0
motion_candidate_cumulative_motion = 0.0
motion_candidate_label = "MOVING OBJECT"
motion_candidate_confidence = 0.50
current_motion_box = None
current_motion_label = ""
last_motion_event_time = 0.0


# ============================================================
# WASTE CLASS CONFIGURATION
# ============================================================

# The custom best.pt model can be trained with any number of waste classes.
# This mapping only normalizes common class-name variations; it does not
# create new classes that the model was not trained to recognize.
WASTE_CLASS_ALIASES = {
    "plastic_bottle": "BOTTLE",
    "plastic bottle": "BOTTLE",
    "bottle": "BOTTLE",
    "plastic": "PLASTIC",
    "paper": "PAPER",
    "cardboard": "CARDBOARD",
    "glass": "GLASS",
    "metal": "METAL",
    "can": "CAN",
    "tin": "CAN",
    "food": "FOOD WASTE",
    "food_waste": "FOOD WASTE",
    "organic": "ORGANIC WASTE",
    "organic_waste": "ORGANIC WASTE",
    "e_waste": "E-WASTE",
    "electronic": "E-WASTE",
    "waste": "WASTE",
}

def normalize_waste_class_name(name):
    raw = str(name).strip()
    key = raw.lower().replace("-", "_")
    return WASTE_CLASS_ALIASES.get(key, raw.upper())

def get_custom_model_classes():
    try:
        names = waste_model.names
        if isinstance(names, dict):
            return {int(k): normalize_waste_class_name(v) for k, v in names.items()}
        return {int(i): normalize_waste_class_name(v) for i, v in enumerate(names)}
    except Exception:
        return {}


# ============================================================
# CAMERA
# ============================================================

camera = None
camera_lock = threading.Lock()

# Browser stream cache: only one thread reads the phone camera.
stream_lock = threading.Lock()
stream_thread = None
stream_stop_event = threading.Event()
stream_running = False
latest_frame_bytes = None
latest_frame_time = 0.0
stream_error = ""


# ============================================================
# LIVE STATUS
# ============================================================

current_person_count = 0
current_waste_count = 0

current_interaction = False
current_holding = False
current_releasing = False
current_illegal_dumping = False
current_legal_disposal = False
current_release_streak = 0
current_missing_waste_streak = 0
current_dumping_reason = ""

current_tracked_person_ids = []
cached_persons = []
cached_wastes = []
recorded_waste_centers = []
recorded_illegal_dumps = {}


# ============================================================
# DUMPING STATE
# ============================================================

holding_streak = 0
release_streak = 0

holding_confirmed = False
release_confirmed = False

holding_person_id = None
holding_waste_center = None
holding_waste_box = None
holding_waste_class = "WASTE"
holding_waste_confidence = 0.0

last_waste_center = None
last_person_center = None
last_wrist_center = None
no_person_release_streak = 0
release_motion_streak = 0

last_dumping_event_time = 0
last_illegal_dumping_time = 0
last_legal_disposal_time = 0
LEGAL_ALERT_VISIBLE_SECONDS = 5

# Person-independent dustbin-zone state.
zone_warmup_counter = 0
zone_empty_streak = 0
zone_entry_streak = 0
zone_last_waste_centers = []
zone_trigger_class = "WASTE"
zone_trigger_confidence = 0.0
zone_last_trigger_time = 0
zone_last_trigger_box = None
zone_candidate_center = None
zone_candidate_streak = 0
zone_candidate_class = "WASTE"
zone_candidate_confidence = 0.0
zone_candidate_box = None
zone_candidate_gap = 0
zone_candidate_start_center = None
zone_candidate_last_center = None
zone_candidate_cumulative_motion = 0.0
personless_empty_streak = 0

last_evidence_image = None


# ============================================================
# SAVE COOLDOWN
# ============================================================

last_person_save_time = 0
last_waste_save_time = 0
last_interaction_save_time = 0


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(
        DATABASE,
        timeout=10
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waste_type TEXT,
            confidence REAL,
            source TEXT,
            detected_at TEXT,
            person_detected INTEGER DEFAULT 0,
            illegal_dumping INTEGER DEFAULT 0,
            interaction_detected INTEGER DEFAULT 0,
            camera_id TEXT DEFAULT 'CAM-01',
            location TEXT DEFAULT 'Unknown',
            evidence_image TEXT,
            person_track_id INTEGER
        )
        """
    )

    conn.commit()
    conn.close()


init_database()


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 60)
print("Loading Waste Detection Model...")
print("=" * 60)

if not os.path.exists(WASTE_MODEL_PATH):
    raise FileNotFoundError(
        f"Waste model not found:\n{WASTE_MODEL_PATH}"
    )

waste_model = YOLO(WASTE_MODEL_PATH)

print("Waste Model Loaded Successfully")

try:
    print("Waste Classes:", waste_model.names)
except Exception:
    pass


print()
print("=" * 60)
print("Loading Person Pose Model...")
print("=" * 60)

if not os.path.exists(POSE_MODEL_PATH):
    print("ERROR: yolo11n-pose.pt not found.")
    print(
        "Run once:"
        "\npython -c "
        "\"from ultralytics import YOLO; "
        "YOLO('yolo11n-pose.pt')\""
    )
    raise FileNotFoundError(POSE_MODEL_PATH)

pose_model = YOLO(POSE_MODEL_PATH)

print("Person Pose Model Loaded Successfully")
print("=" * 60)
print()

print("Loading Bottle Detection Model...")
try:
    bottle_model = YOLO(BOTTLE_MODEL_PATH)
    print("Bottle Detection Model Loaded Successfully")
except Exception as e:
    print("Bottle model load failed:", e)
    print("Ultralytics will try to download yolo11n.pt automatically when used.")
    bottle_model = YOLO("yolo11n.pt")
    print("Bottle Detection Model Ready")
print()


# ============================================================
# DATABASE SAVE
# ============================================================

def save_detection(
    waste_type=None,
    confidence=0.0,
    source="camera",
    person_detected=False,
    illegal_dumping=False,
    interaction_detected=False,
    evidence_image=None,
    person_track_id=None
):

    conn = None

    try:

        conn = get_db()

        conn.execute(
            """
            INSERT INTO detections (
                waste_type,
                confidence,
                source,
                detected_at,
                person_detected,
                illegal_dumping,
                interaction_detected,
                camera_id,
                location,
                evidence_image,
                person_track_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                waste_type,
                float(confidence),
                source,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                int(person_detected),
                int(illegal_dumping),
                int(interaction_detected),
                CAMERA_ID,
                CAMERA_LOCATION,
                evidence_image,
                person_track_id
            )
        )

        conn.commit()

        return True

    except Exception as e:

        print(
            "Database save error:",
            e
        )

        return False

    finally:

        if conn is not None:
            conn.close()


# ============================================================
# GEOMETRY
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def calculate_distance(
    point1,
    point2
):

    if point1 is None or point2 is None:
        return 999999

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    return (
        (dx * dx + dy * dy)
        ** 0.5
    )


# ============================================================
# PERSON DETECTION + POSE
# ============================================================

def detect_persons(frame):

    persons = []

    try:

        results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=PERSON_CONFIDENCE,
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False
        )

        if not results:
            return persons

        result = results[0]

        if result.boxes is None:
            return persons

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .tolist()
            )

            confidence = float(
                boxes.conf[i]
                .cpu()
                .item()
            )

            pw = int(xyxy[2] - xyxy[0])
            ph = int(xyxy[3] - xyxy[1])
            if pw * ph < 5000 or pw < 45 or ph < 75:
                continue

            track_id = None

            if boxes.id is not None:

                try:
                    track_id = int(
                        boxes.id[i]
                        .cpu()
                        .item()
                    )
                except Exception:
                    track_id = None

            wrists = []
            elbows = []

            if result.keypoints is not None:

                try:

                    keypoints = (
                        result.keypoints.xy[i]
                        .cpu()
                        .numpy()
                    )

                    if len(keypoints) >= 11:

                        left_elbow = keypoints[7]
                        right_elbow = keypoints[8]
                        left_wrist = keypoints[9]
                        right_wrist = keypoints[10]

                        if (
                            left_elbow[0] > 0
                            and left_elbow[1] > 0
                        ):
                            elbows.append((int(left_elbow[0]), int(left_elbow[1])))

                        if (
                            right_elbow[0] > 0
                            and right_elbow[1] > 0
                        ):
                            elbows.append((int(right_elbow[0]), int(right_elbow[1])))

                        if (
                            left_wrist[0] > 0
                            and left_wrist[1] > 0
                        ):
                            wrists.append(
                                (
                                    int(left_wrist[0]),
                                    int(left_wrist[1])
                                )
                            )

                        if (
                            right_wrist[0] > 0
                            and right_wrist[1] > 0
                        ):
                            wrists.append(
                                (
                                    int(right_wrist[0]),
                                    int(right_wrist[1])
                                )
                            )

                except Exception:
                    wrists = []

            persons.append(
                {
                    "box": [
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3])
                    ],
                    "confidence": confidence,
                    "track_id": track_id,
                    "wrists": wrists,
                    "elbows": elbows
                }
            )

    except Exception as e:

        print(
            "Person detection error:",
            e
        )

    return persons


# ============================================================
# WASTE DETECTION
# ============================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def detect_waste(frame):
    wastes = []

    # --------------------------------------------------------
    # 1) Custom waste model
    # --------------------------------------------------------
    try:
        results = waste_model.predict(
            frame,
            conf=WASTE_CONFIDENCE,
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False
        )

        if results:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    xyxy = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    confidence = float(
                        boxes.conf[i]
                        .cpu()
                        .item()
                    )

                    class_id = int(
                        boxes.cls[i]
                        .cpu()
                        .item()
                    )

                    try:
                        class_name = normalize_waste_class_name(
                            waste_model.names[class_id]
                        )
                    except Exception:
                        class_name = "WASTE"

                    wastes.append({
                        "box": [
                            int(xyxy[0]),
                            int(xyxy[1]),
                            int(xyxy[2]),
                            int(xyxy[3])
                        ],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "source": "custom_waste"
                    })

    except Exception as e:
        print("Waste detection error:", e)

    # --------------------------------------------------------
    # 2) COCO bottle and cup detection (plastic containers)
    # COCO class 39 = bottle, class 40 = wine glass/bottle, class 41 = cup, class 45 = bowl (containers)
    # --------------------------------------------------------
    try:
        results = bottle_model.predict(
            frame,
            conf=BOTTLE_CONFIDENCE,
            classes=[39, 40, 41, 45],
            imgsz=INFERENCE_IMAGE_SIZE,
            verbose=False
        )

        if results:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    xyxy = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    confidence = float(
                        boxes.conf[i]
                        .cpu()
                        .item()
                    )

                    cid = int(
                        boxes.cls[i]
                        .cpu()
                        .item()
                    )
                    if cid == 39:
                        cname = "BOTTLE"
                    elif cid == 40:
                        cname = "BOTTLE / GLASS"
                    elif cid == 41:
                        cname = "PLASTIC CUP"
                    elif cid == 45:
                        cname = "PLASTIC CONTAINER"
                    else:
                        cname = "PLASTIC / WASTE"

                    # Avoid duplicate boxes when the custom
                    # waste model already detects the same object.
                    obj_box = [
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3])
                    ]

                    duplicate = False
                    for existing in wastes:
                        if calculate_iou(
                            obj_box,
                            existing["box"]
                        ) >= 0.50:
                            duplicate = True
                            if confidence > existing["confidence"]:
                                existing["confidence"] = confidence
                                existing["class_name"] = cname
                                existing["source"] = "coco_detector"
                            break

                    if not duplicate:
                        wastes.append({
                            "box": obj_box,
                            "confidence": confidence,
                            "class_id": cid,
                            "class_name": cname,
                            "source": "coco_detector"
                        })

    except Exception as e:
        print("COCO bottle/cup detection error:", e)

    return wastes


# ============================================================
# BODY / WASTE FALSE-POSITIVE GUARDS
# ============================================================

def intersection_over_waste(box_a, box_b):
    """Return fraction of box_a covered by box_b."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area


def point_to_segment_distance(point, seg_a, seg_b):
    if point is None or seg_a is None or seg_b is None:
        return 999999.0
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return calculate_distance(point, seg_a)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx, cy = ax + t * vx, ay + t * vy
    return calculate_distance(point, (cx, cy))


def waste_likely_body_part(person, waste):
    """Reject generic-waste boxes that look like clothing/skin/body parts.

    A real held object should have a meaningful portion of its box outside
    the person bounding box and should not be centered along the forearm.
    """
    name = str(waste.get("class_name", "waste")).strip().lower()
    if name not in {"waste", "generic waste"}:
        return False

    pbox = person.get("box")
    wbox = waste.get("box")
    if not pbox or not wbox:
        return False

    iou = box_iou(wbox, pbox)
    inside_fraction = intersection_over_waste(wbox, pbox)
    wx, wy = box_center(wbox)
    center_inside = (pbox[0] <= wx <= pbox[2] and pbox[1] <= wy <= pbox[3])

    # If most of the detected object is physically inside the person box,
    # generic WASTE is much more likely to be the shirt/pants/skin.
    if inside_fraction >= 0.60 and center_inside:
        return True

    # A generic box centered on the elbow->wrist segment is very likely the arm.
    elbows = person.get("elbows", [])
    wrists = person.get("wrists", [])
    for elbow in elbows:
        for wrist in wrists:
            arm_len = calculate_distance(elbow, wrist)
            if arm_len <= 5:
                continue
            arm_margin = max(25.0, min(55.0, arm_len * 0.35))
            arm_dist = point_to_segment_distance((wx, wy), elbow, wrist)
            if arm_dist <= arm_margin and inside_fraction >= 0.25:
                return True

    # Large generic regions overlapping the person and mostly contained in the
    # body rectangle are also treated as body false positives.
    p_area = max(1, box_area(pbox))
    w_area = box_area(wbox)
    relative = w_area / p_area
    if iou >= 0.12 and relative >= 0.08 and inside_fraction >= 0.45:
        return True

    return False


# ============================================================
# HAND / WASTE
# ============================================================

def waste_near_hand(
    person,
    waste,
    threshold=HAND_WASTE_DISTANCE
):

    wrists = person.get(
        "wrists",
        []
    )

    if not wrists:
        return False

    waste_center = box_center(
        waste["box"]
    )

    for wrist in wrists:

        d = calculate_distance(
            wrist,
            waste_center
        )

        if d <= threshold:
            return True

    return False


def valid_waste_for_holding(person, waste):
    """Ensure ANY waste (bottle, paper, trash) is genuinely held near the wrist or hand."""
    wbox = waste.get("box")
    pbox = person.get("box")
    if not wbox or not pbox:
        return False

    wcenter = box_center(wbox)
    wrists = person.get("wrists", [])

    # 1. Holding MUST have wrist proximity (person actually holding with hands)
    if not wrists:
        return False
    near_wrist = any(calculate_distance(wrist, wcenter) <= HAND_WASTE_DISTANCE for wrist in wrists)
    if not near_wrist:
        return False

    # 2. Reject body / clothing false positives
    name = str(waste.get("class_name", "waste")).strip().lower()
    if name in {"waste", "generic waste"}:
        if waste_likely_body_part(person, waste):
            return False
        inside_fraction = intersection_over_waste(wbox, pbox)
        if inside_fraction >= 0.65:
            return False

    return True


def find_holding_pair(
    persons,
    wastes
):

    for person in persons:

        for waste in wastes:

            if valid_waste_for_holding(person, waste):
                return (
                    person,
                    waste
                )

    return None, None


# ============================================================
# RESET
# ============================================================

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# AUTOMATIC DUSTBIN RECEPTACLE DETECTION (BUCKET / CONTAINER)
# ============================================================
auto_detected_dustbin = None
auto_detected_dustbin_smooth = None
dustbin_missing_frames = 0

def detect_dustbin_receptacle(frame):
    """Dynamically detects the physical dustbin / bucket receptacle and lid in the frame.
    Supports the user's red bucket dustbin automatically via precise HSV color segmentation.
    Accurately isolates ONLY the physical bucket receptacle without expanding into floor tiles or waste.
    """
    global auto_detected_dustbin, auto_detected_dustbin_smooth, dustbin_missing_frames

    if frame is None or frame.size == 0:
        return [int(round(v)) for v in auto_detected_dustbin_smooth] if auto_detected_dustbin_smooth else auto_detected_dustbin

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # True red plastic saturation: require S >= 75 so floor tiles (S ~ 40) are completely excluded!
    m1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 75, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # Exclude rightmost 10% where wall curtain hangs
    red_mask[:, int(width * 0.90):] = 0
    red_mask[:4, :] = 0
    red_mask[max(0, height - 4):, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            solidity = area / float(w * h)
            ar = float(h) / float(w)

            # Reject tall thin vertical strips (curtain edges)
            if ar > 2.5 or solidity < 0.40:
                continue

            if 0.35 <= ar <= 2.2 and solidity >= 0.40 and w >= 40 and h >= 40:
                score = area * solidity
                candidates.append((score, (x, y, x + w, y + h), (x, y, w, h)))

    if not candidates:
        dustbin_missing_frames += 1
        if dustbin_missing_frames > 40:
            auto_detected_dustbin_smooth = None
            auto_detected_dustbin = None
        return [int(round(v)) for v in auto_detected_dustbin_smooth] if auto_detected_dustbin_smooth else auto_detected_dustbin

    dustbin_missing_frames = 0

    # Pick the best candidates and merge adjacent bucket + lid within 35px
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_box = list(candidates[0][1])

    for other_score, other_box, _ in candidates[1:]:
        ox1, oy1, ox2, oy2 = other_box
        dist_x = max(0, max(best_box[0] - ox2, ox1 - best_box[2]))
        dist_y = max(0, max(best_box[1] - oy2, oy1 - best_box[3]))
        if dist_x <= 35 and dist_y <= 35:
            best_box = [
                min(best_box[0], ox1),
                min(best_box[1], oy1),
                max(best_box[2], ox2),
                max(best_box[3], oy2)
            ]

    pad = 5
    final_box = [
        max(0, int(best_box[0] - pad)),
        max(0, int(best_box[1] - pad)),
        min(width, int(best_box[2] + pad)),
        min(height, int(best_box[3] + pad))
    ]

    auto_detected_dustbin = final_box
    if auto_detected_dustbin_smooth is None:
        auto_detected_dustbin_smooth = [float(v) for v in final_box]
    else:
        # Smooth EMA (75% old, 25% new) for stable, jitter-free box
        for i in range(4):
            auto_detected_dustbin_smooth[i] = (
                0.75 * auto_detected_dustbin_smooth[i] + 0.25 * final_box[i]
            )

    return [int(round(v)) for v in auto_detected_dustbin_smooth]


def get_effective_dustbin_zone(frame=None):
    """Return the active dustbin zone rectangle [x1, y1, x2, y2].

    Prioritizes auto-detection of the physical receptacle (bucket) if available.
    Falls back to DUSTBIN_ZONE if auto-detection is not yet confident.
    """
    global auto_detected_dustbin, auto_detected_dustbin_smooth
    if frame is not None:
        detected = detect_dustbin_receptacle(frame)
        if detected is not None:
            return detected
    if auto_detected_dustbin_smooth is not None:
        return [int(round(v)) for v in auto_detected_dustbin_smooth]
    if auto_detected_dustbin is not None:
        return auto_detected_dustbin
    if DUSTBIN_ZONE_ENABLED and DUSTBIN_ZONE is not None:
        return DUSTBIN_ZONE
    return None


def is_waste_inside_dustbin(waste_or_center, frame=None):
    """Check if a waste object or coordinate is inside or deposited at the dustbin."""
    if not DUSTBIN_ZONE_ENABLED:
        return False

    db = get_effective_dustbin_zone(frame)
    if db is None:
        return False

    if isinstance(waste_or_center, dict):
        box = waste_or_center.get("box")
        if not box or len(box) != 4:
            return False
        center = box_center(box)
        # 1. Center of waste is inside dustbin zone
        if point_in_rect(center, db):
            return True
        # 2. Or majority (>= 55%) of the waste bounding box is inside the dustbin zone
        overlap = intersection_over_waste(box, db)
        if overlap >= 0.55:
            return True
        return False
    elif isinstance(waste_or_center, (list, tuple)):
        if len(waste_or_center) == 2:
            return point_in_rect(waste_or_center, db)
        elif len(waste_or_center) == 4:
            center = box_center(waste_or_center)
            if point_in_rect(center, db):
                return True
            overlap = intersection_over_waste(waste_or_center, db)
            return overlap >= 0.55
    return False


def is_waste_approaching_dustbin(waste_or_box, frame=None, proximity=50):
    """Check if waste (or hand carrying waste) is reaching into, touching, or adjacent to the dustbin opening.
    When a person reaches to throw trash into the dustbin, it is LEGAL DISPOSAL in progress!
    """
    db = get_effective_dustbin_zone(frame)
    if db is None:
        return False

    box = waste_or_box.get("box") if isinstance(waste_or_box, dict) else waste_or_box
    if not box or len(box) != 4:
        return False

    bx1, by1, bx2, by2 = [int(v) for v in box]
    dx1, dy1, dx2, dy2 = [int(v) for v in db]

    # 1. Overlaps with dustbin opening
    ix1 = max(bx1, dx1)
    iy1 = max(by1, dy1)
    ix2 = min(bx2, dx2)
    iy2 = min(by2, dy2)
    if ix2 > ix1 and iy2 > iy1:
        return True

    # 2. Within proximity of dustbin rim
    dist_x = max(0, max(dx1 - bx2, bx1 - dx2))
    dist_y = max(0, max(dy1 - by2, by1 - dy2))
    if dist_x <= proximity and dist_y <= proximity:
        return True

    return False


def filter_false_waste_detections(wastes, persons, frame=None):
    """Remove dustbin receptacle (bucket/lid), wall curtain, and person body/clothing false positives.
    The dustbin is the designated legal receptacle and is NEVER illegal waste!
    Genuine waste (bottles, papers, wrappers) outside the dustbin is ALWAYS kept!
    """
    if not wastes:
        return []

    dustbin_box = get_effective_dustbin_zone(frame) if frame is not None else None
    height, width = frame.shape[:2] if frame is not None else (480, 640)

    filtered = []
    for waste in wastes:
        box = waste.get("box")
        if not box or len(box) != 4:
            continue

        name = str(waste.get("class_name", "waste")).strip().lower()
        source = str(waste.get("source", ""))
        bx1, by1, bx2, by2 = [int(v) for v in box]
        bw = bx2 - bx1
        bh = by2 - by1

        # ----------------------------------------------------
        # 1) FILTER DUSTBIN RECEPTACLE & LID (NEVER WASTE!)
        # ----------------------------------------------------
        if source != "coco_detector" and name in {"waste", "generic waste"}:
            # A) Overlaps significantly with the detected physical dustbin
            # Require high overlap (IoU >= 0.40 or box almost completely inside IoW >= 0.70)
            # so real waste sitting outside/near the dustbin is NEVER filtered out!
            if dustbin_box is not None:
                iou_with_db = box_iou(box, dustbin_box)
                cov_by_db = intersection_over_waste(box, dustbin_box)
                if iou_with_db >= 0.40 or cov_by_db >= 0.70:
                    continue

            # B) Direct check for true red plastic bucket body / lid (S >= 75, red_ratio >= 0.35)
            # Bucket is ~88% red, lid is ~78% red. Real waste (packets, paper) is < 5% red!
            if frame is not None and frame.size > 0 and bw >= 45 and bh >= 45:
                crop = frame[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)]
                if crop.size > 0:
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    cm1 = cv2.inRange(crop_hsv, np.array([0, 75, 40]), np.array([15, 255, 255]))
                    cm2 = cv2.inRange(crop_hsv, np.array([160, 75, 40]), np.array([180, 255, 255]))
                    red_ratio = float(((cm1 > 0) | (cm2 > 0)).mean())
                    # If this box is vibrant red plastic (bucket or lid >= 35%), it is the dustbin!
                    if red_ratio >= 0.35:
                        continue

            # C) Filter background wall curtain on extreme right edge
            if bx2 >= width - 50 and bh >= int(height * 0.55):
                continue

            # D) Filter reaching arm/hand limb false positives
            # An arm entering from the camera border to reach into or near the dustbin is a person limb, not trash!
            is_large_limb = (bw >= 180 or bh >= 160 or (bw * bh >= 28000))
            touches_border = (bx1 <= 15 or bx2 >= width - 15 or by2 >= height - 15)
            if touches_border and is_large_limb and is_waste_approaching_dustbin(box, frame, proximity=40):
                continue

        # ----------------------------------------------------
        # 2) FILTER PERSON BODY / CLOTHING FALSE POSITIVES
        # ----------------------------------------------------
        keep = True
        if persons:
            for person in persons:
                if name in {"waste", "generic waste"}:
                    if waste_likely_body_part(person, waste):
                        keep = False
                        break

                    inside_fraction = intersection_over_waste(box, person["box"])
                    iou = box_iou(box, person["box"])
                    outside_fraction = 1.0 - inside_fraction
                    near_hand = waste_near_hand(
                        person, waste, threshold=max(HAND_WASTE_DISTANCE, 120)
                    )

                    if (iou >= 0.08 or inside_fraction >= 0.30) and not (
                        near_hand and outside_fraction >= 0.30
                    ):
                        keep = False
                        break

        if keep:
            filtered.append(waste)

    return filtered


def reset_dumping_state():

    global holding_streak
    global release_streak

    global holding_confirmed
    global release_confirmed

    global holding_person_id
    global holding_waste_center
    global holding_waste_box
    global holding_waste_class
    global holding_waste_confidence

    global last_waste_center
    global last_person_center
    global last_wrist_center
    global no_person_release_streak
    global release_motion_streak

    global current_interaction
    global current_holding
    global current_releasing

    holding_streak = 0
    release_streak = 0

    holding_confirmed = False
    release_confirmed = False

    holding_person_id = None
    holding_waste_center = None
    holding_waste_box = None
    holding_waste_class = "WASTE"
    holding_waste_confidence = 0.0

    last_waste_center = None
    last_person_center = None
    last_wrist_center = None
    no_person_release_streak = 0
    release_motion_streak = 0

    current_interaction = False
    current_holding = False
    current_releasing = False


# ============================================================
# EVIDENCE IMAGE
# ============================================================

def create_evidence_image(
    frame,
    persons,
    wastes
):

    global last_evidence_image

    try:

        evidence = frame.copy()

        height, width = evidence.shape[:2]

        cv2.rectangle(
            evidence,
            (0, 0),
            (width, 36),
            (0, 0, 200),
            -1
        )

        header_text = f"ILLEGAL DUMPING (OUTSIDE DUSTBIN) | {CAMERA_ID} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(
            evidence,
            header_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        # Draw dustbin zone if detected
        db_zone = get_effective_dustbin_zone(frame)
        if db_zone is not None:
            dx1, dy1, dx2, dy2 = [int(v) for v in db_zone]
            cv2.rectangle(evidence, (dx1, dy1), (dx2, dy2), (0, 200, 0), 2)
            cv2.putText(
                evidence,
                "DUSTBIN (LEGAL RECEPTACLE)",
                (dx1, max(dy1 - 8, 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                2
            )

        for person in persons:

            x1, y1, x2, y2 = [int(v) for v in person["box"]]

            track_id = person.get(
                "track_id"
            )

            cv2.rectangle(
                evidence,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                evidence,
                f"PERSON ID:{track_id}",
                (x1, max(y1 - 8, 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                2
            )

        for waste in wastes:

            x1, y1, x2, y2 = [int(v) for v in waste["box"]]

            cv2.rectangle(
                evidence,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                evidence,
                f"ILLEGAL: {waste.get('class_name', 'WASTE')} {float(waste.get('confidence', 0.0)):.2f} [OUTSIDE DUSTBIN]",
                (x1, max(y1 - 8, 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2
            )

        filename = (
            "illegal_dumping_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".jpg"
        )

        # Always make sure the evidence directory exists.
        os.makedirs(RESULT_FOLDER, exist_ok=True)

        filepath = os.path.join(
            RESULT_FOLDER,
            filename
        )

        # Write the evidence image and verify that the file really exists
        # before storing its URL in SQLite. This prevents history records
        # from pointing to an evidence file that was never written.
        write_ok = cv2.imwrite(
            filepath,
            evidence
        )

        if (
            not write_ok
            or not os.path.isfile(filepath)
            or os.path.getsize(filepath) == 0
        ):
            print(
                "Evidence image write failed:",
                filepath
            )
            return None

        # Store the Flask evidence route, not a static path.
        # The route sanitizes the filename and serves the exact file.
        last_evidence_image = (
            f"/evidence/{filename}"
        )

        print(
            "Evidence saved and verified:",
            filepath
        )

        return last_evidence_image

    except Exception as e:

        print(
            "Evidence creation error:",
            e
        )

        return None


# ============================================================
# PROCESS DUMPING
# ============================================================

def process_dumping(
    persons,
    wastes,
    frame
):

    global holding_streak
    global release_streak

    global holding_confirmed
    global release_confirmed

    global holding_person_id
    global holding_waste_center
    global holding_waste_box
    global holding_waste_class
    global holding_waste_confidence

    global last_waste_center
    global last_person_center
    global last_wrist_center
    global no_person_release_streak
    global release_motion_streak

    global last_dumping_event_time
    global last_illegal_dumping_time
    global last_legal_disposal_time

    global current_interaction
    global current_holding
    global current_releasing
    global current_illegal_dumping
    global current_legal_disposal
    global current_release_streak
    global current_missing_waste_streak
    global current_dumping_reason

    current_interaction = False
    current_holding = False
    current_releasing = False
    current_release_streak = release_streak
    current_missing_waste_streak = 0
    current_dumping_reason = ""

    # --------------------------------------------------------
    # BEFORE HOLDING: normal waste/person detection only
    # --------------------------------------------------------
    if not holding_confirmed:
        holding_person, holding_waste = find_holding_pair(persons, wastes) if wastes else (None, None)

        if holding_person is not None and holding_waste is not None:
            current_interaction = True
            current_holding = True
            release_streak = 0
            release_confirmed = False
            release_motion_streak = 0

            person_id = holding_person.get("track_id")
            waste_center = box_center(holding_waste["box"])

            holding_streak += 1

            if holding_streak >= REQUIRED_HOLDING_FRAMES:
                holding_confirmed = True
                holding_person_id = person_id
                holding_waste_center = waste_center
                holding_waste_box = list(holding_waste["box"])
                holding_waste_class = holding_waste.get("class_name", "WASTE")
                holding_waste_confidence = float(holding_waste.get("confidence", 0.0))

                last_waste_center = waste_center
                last_person_center = box_center(holding_person["box"])
                wrists = holding_person.get("wrists", [])
                last_wrist_center = (
                    min(wrists, key=lambda w: calculate_distance(w, waste_center))
                    if wrists else None
                )
                no_person_release_streak = 0

                print()
                print("=" * 60)
                print("WASTE HOLDING CONFIRMED")
                print("Person ID:", person_id)
                print("Waste:", holding_waste_class)
                print("Holding center:", holding_waste_center)
                print("=" * 60)
        else:
            holding_streak = 0
            current_holding = False
            current_releasing = False

        current_release_streak = release_streak
        return False

    # --------------------------------------------------------
    # AFTER HOLDING: only the SAME held object can trigger release
    # --------------------------------------------------------
    selected_person = None
    for person in persons:
        if person.get("track_id") == holding_person_id:
            selected_person = person
            break

    # Important: a confirmed holding state should not instantly become an
    # illegal-dumping event if the person simply leaves the frame.
    if selected_person is None:
        no_person_release_streak += 1
        current_releasing = False
        current_holding = False
        current_dumping_reason = "PERSON TRACK LOST"

        if no_person_release_streak > NO_PERSON_RELEASE_GRACE_FRAMES:
            reset_dumping_state()
            current_dumping_reason = "RESET: PERSON TRACK LOST"

        current_release_streak = release_streak
        return False

    no_person_release_streak = 0
    current_person_center = box_center(selected_person["box"])
    wrists = selected_person.get("wrists", [])

    # --------------------------------------------------------
    # Associate current waste with the held waste.
    # We deliberately do NOT use person-body center distance as a release
    # trigger; only wrist separation + object motion / disappearance counts.
    # --------------------------------------------------------
    associated_waste = None

    if wastes and holding_waste_center is not None:
        nearest_distance = None
        for waste in wastes:
            d = calculate_distance(
                box_center(waste["box"]),
                holding_waste_center
            )
            if nearest_distance is None or d < nearest_distance:
                nearest_distance = d
                associated_waste = waste

        if nearest_distance is not None and nearest_distance > MAX_HOLDING_ASSOCIATION_DISTANCE:
            associated_waste = None

        if associated_waste is not None and str(associated_waste.get("class_name", "waste")).strip().lower() in {"waste", "generic waste"}:
            if not valid_waste_for_holding(selected_person, associated_waste):
                associated_waste = None

    # --------------------------------------------------------
    # CASE A: held waste is still visible
    # --------------------------------------------------------
    if associated_waste is not None:
        current_waste_center = box_center(associated_waste["box"])
        current_waste_box = list(associated_waste["box"])
        current_waste_class = associated_waste.get("class_name", holding_waste_class)
        current_waste_confidence = float(
            associated_waste.get("confidence", holding_waste_confidence)
        )

        wrist_distance = 999999.0
        if wrists:
            wrist_distance = min(
                calculate_distance(wrist, current_waste_center)
                for wrist in wrists
            )

        motion_distance = 0.0
        if last_waste_center is not None:
            motion_distance = calculate_distance(
                current_waste_center,
                last_waste_center
            )

        # True dumping requires:
        # 1. Clear separation from wrist (distance >= RELEASE_DISTANCE)
        # 2. Movement downward toward ground (y increases) or landing near floor level
        moved_from_hold = calculate_distance(
            current_waste_center,
            holding_waste_center
        ) if holding_waste_center is not None else 0.0

        far_from_wrist = (wrist_distance >= RELEASE_DISTANCE) or (wrist_distance >= HAND_WASTE_DISTANCE + 15)
        moved_enough = motion_distance >= MIN_RELEASE_MOTION or moved_from_hold >= MIN_RELEASE_MOTION

        # Vertical check: downward drop towards floor (in pixel coordinates, y increases downwards)
        downward_movement = False
        if holding_waste_center is not None:
            dy = current_waste_center[1] - holding_waste_center[1]
            downward_movement = dy >= MIN_DOWNWARD_DROP_MOTION

        # Low position check: object is clearly below the person's midsection (near floor)
        near_floor_level = (
            current_person_center is not None
            and current_waste_center[1] > current_person_center[1] + 25
        )

        # Horizontal or lateral throw: object displaced significantly from hold position
        horizontal_throw = moved_from_hold >= 35.0

        is_dropping_or_floor = downward_movement or near_floor_level or horizontal_throw

        if far_from_wrist and is_dropping_or_floor and moved_enough:
            release_streak += 1
            release_motion_streak += 1
            current_releasing = True
            current_holding = False
            current_dumping_reason = (
                f"WRIST SEPARATED: {wrist_distance:.1f}px | "
                f"DROP/THROW CONFIRMED | "
                f"RELEASE: {release_streak}/{REQUIRED_RELEASE_FRAMES}"
            )
        else:
            # Still held, drinking, or lifting: decay release streak
            release_streak = max(0, release_streak - 1)
            release_motion_streak = max(0, release_motion_streak - 1)
            current_releasing = release_streak > 0
            current_holding = wrist_distance <= HAND_WASTE_DISTANCE
            current_dumping_reason = (
                f"WRIST DIST: {wrist_distance:.1f}px | "
                f"HOLD: {current_holding}"
            )

        last_waste_center = current_waste_center
        last_person_center = current_person_center
        holding_waste_box = current_waste_box
        holding_waste_class = current_waste_class
        holding_waste_confidence = current_waste_confidence

        if release_streak >= REQUIRED_RELEASE_FRAMES:
            release_confirmed = True

    # --------------------------------------------------------
    # CASE B: held waste disappeared / occluded
    # --------------------------------------------------------
    else:
        current_missing_waste_streak += 1
        
        # Only treat disappearance as an illegal throw if the waste was ALREADY
        # in active downward/falling motion before leaving view!
        was_actively_dropping = (
            last_waste_center is not None
            and holding_waste_center is not None
            and (
                (last_waste_center[1] - holding_waste_center[1] >= MIN_DOWNWARD_DROP_MOTION)
                or calculate_distance(last_waste_center, holding_waste_center) >= 30.0
            )
            and (release_motion_streak >= 1 or release_streak >= 1)
        )

        # Disappeared without landing: do not trigger false throw
        current_releasing = False
        current_holding = False
        current_dumping_reason = "OBJECT OCCLUDED / LOST"
        if current_missing_waste_streak >= MISSING_WASTE_RELEASE_FRAMES:
            reset_dumping_state()

    current_release_streak = release_streak

    # --------------------------------------------------------
    # CONFIRMED DUMPING EVALUATION (DUSTBIN VS OUTSIDE DUSTBIN)
    # --------------------------------------------------------
    if release_confirmed:
        now = time.time()

        if now - last_dumping_event_time < DUMPING_EVENT_COOLDOWN:
            reset_dumping_state()
            return False

        # Determine waste landing / dropped location
        landing_point = None
        if associated_waste is not None:
            landing_point = box_center(associated_waste["box"])
        elif last_waste_center is not None:
            landing_point = last_waste_center
        elif holding_waste_center is not None:
            landing_point = holding_waste_center

        # Check if landing position falls inside the designated Dustbin
        is_inside_dustbin = is_waste_inside_dustbin(landing_point, frame)

        evidence_wastes = []
        if associated_waste is not None:
            evidence_wastes.append(associated_waste)
        elif holding_waste_box is not None:
            evidence_wastes.append({
                "box": list(holding_waste_box),
                "confidence": float(holding_waste_confidence),
                "class_name": holding_waste_class,
                "class_id": 0,
                "source": "held_waste"
            })

        if is_inside_dustbin:
            # ====================================================
            # CASE 1: PROPER DISPOSAL (LEGAL - INSIDE DUSTBIN)
            # DO NOT trigger illegal dumping alert!
            # ====================================================
            current_illegal_dumping = False
            current_legal_disposal = True
            last_legal_disposal_time = now
            last_dumping_event_time = now
            current_dumping_reason = (
                f"PROPER DISPOSAL: Waste deposited INSIDE Dustbin! "
                f"(Landing: {landing_point})"
            )

            # Record legal disposal in database
            for waste in evidence_wastes:
                save_detection(
                    waste_type=waste.get("class_name", holding_waste_class),
                    confidence=waste.get("confidence", holding_waste_confidence),
                    source="camera",
                    person_detected=True,
                    illegal_dumping=False,
                    interaction_detected=True,
                    evidence_image=None,
                    person_track_id=holding_person_id
                )

            print()
            print("=" * 60)
            print("🟢 PROPER DISPOSAL CONFIRMED (LEGAL - INSIDE DUSTBIN)")
            print("Person ID:", holding_person_id)
            print("Waste:", holding_waste_class)
            print("Landing Point:", landing_point)
            print("Dustbin:", get_effective_dustbin_zone(frame))
            print("Status: LEGAL (No alert triggered)")
            print("=" * 60)

            reset_dumping_state()
            return False

        # ====================================================
        # CASE 2: ILLEGAL DUMPING (THROWN OUTSIDE DUSTBIN)
        # ====================================================
        evidence_image = create_evidence_image(
            frame,
            persons,
            evidence_wastes
        )

        saved_any = False
        for waste in evidence_wastes:
            saved = save_detection(
                waste_type=waste.get("class_name", holding_waste_class),
                confidence=waste.get("confidence", holding_waste_confidence),
                source="camera",
                person_detected=True,
                illegal_dumping=True,
                interaction_detected=True,
                evidence_image=evidence_image,
                person_track_id=holding_person_id
            )
            saved_any = saved_any or saved

        last_dumping_event_time = now
        last_illegal_dumping_time = now
        current_illegal_dumping = True
        current_legal_disposal = False
        current_dumping_reason = (
            f"ILLEGAL DUMPING: Waste thrown OUTSIDE Dustbin! "
            f"(Landing: {landing_point})"
        )

        print()
        print("=" * 60)
        print("🚨 ILLEGAL DUMPING CONFIRMED (OUTSIDE DUSTBIN)")
        print("Person ID:", holding_person_id)
        print("Waste:", holding_waste_class)
        print("Landing Point:", landing_point)
        print("Dustbin Zone:", DUSTBIN_ZONE)
        print("Reason:", current_dumping_reason)
        print("Evidence:", evidence_image)
        print("Database saved:", saved_any)
        print("=" * 60)

        reset_dumping_state()
        return True

    return False


# ============================================================
# PERSON-INDEPENDENT DUSTBIN / DUMPING-ZONE DETECTION
# ============================================================

def point_in_rect(point, rect):
    if point is None or rect is None:
        return False
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def process_zone_dumping(wastes, frame, person_present=False):
    """Strict person-independent dumping detection.

    Only runs when no person is visible. A stationary object or a detector
    jitter must NOT create an incident. A candidate must be a reasonably
    confident waste detection that shows real cumulative motion across
    multiple inference frames before an incident is confirmed.
    """
    global zone_warmup_counter
    global zone_last_waste_centers
    global zone_entry_streak
    global zone_trigger_class
    global zone_trigger_confidence
    global zone_last_trigger_time
    global last_dumping_event_time
    global last_illegal_dumping_time
    global last_legal_disposal_time
    global current_illegal_dumping
    global current_legal_disposal
    global current_dumping_reason
    global holding_confirmed
    global zone_candidate_center
    global zone_candidate_streak
    global zone_candidate_class
    global zone_candidate_confidence
    global zone_candidate_box
    global zone_candidate_gap
    global zone_candidate_start_center
    global zone_candidate_last_center
    global zone_candidate_cumulative_motion

    if not DUSTBIN_ZONE_ENABLED:
        return False

    # Pause personless logic only if person is already actively holding waste
    if person_present and holding_confirmed:
        zone_entry_streak = 0
        zone_candidate_center = None
        zone_candidate_streak = 0
        zone_candidate_gap = 0
        zone_candidate_start_center = None
        zone_candidate_last_center = None
        zone_candidate_cumulative_motion = 0.0
        zone_last_waste_centers = [
            (box_center(w["box"]), w.get("class_name", "WASTE"))
            for w in wastes
            if float(w.get("confidence", 0.0)) >= PERSONLESS_MIN_CONFIDENCE
        ]
        current_dumping_reason = "PERSON HOLDING - PERSONLESS TRACKING PAUSED"
        return False

    current_wastes = [
        w for w in wastes
        if float(w.get("confidence", 0.0)) >= PERSONLESS_MIN_CONFIDENCE
    ]

    # Warm-up establishes the scene baseline. Nothing already present is an
    # illegal-dumping event merely because it is visible.
    if zone_warmup_counter < DUSTBIN_ZONE_WARMUP_FRAMES:
        zone_warmup_counter += 1
        zone_last_waste_centers = [
            (box_center(w["box"]), w.get("class_name", "WASTE"))
            for w in current_wastes
        ]
        zone_entry_streak = 0
        zone_candidate_center = None
        zone_candidate_streak = 0
        zone_candidate_gap = 0
        zone_candidate_start_center = None
        zone_candidate_last_center = None
        zone_candidate_cumulative_motion = 0.0
        current_dumping_reason = (
            f"PERSONLESS WARMUP {zone_warmup_counter}/{DUSTBIN_ZONE_WARMUP_FRAMES}"
        )
        return False

    # After every confirmed event, require a longer cooldown before a new
    # personless event can be generated.
    now = time.time()
    if now - zone_last_trigger_time < PERSONLESS_EVENT_COOLDOWN:
        zone_last_waste_centers = [
            (box_center(w["box"]), w.get("class_name", "WASTE"))
            for w in current_wastes
        ]
        current_dumping_reason = "PERSONLESS COOLDOWN"
        return False

    candidate = None
    candidate_frame_motion = 0.0

    # Match candidates to the previous baseline. A new object is allowed only
    # as a candidate; it still must demonstrate actual trajectory movement.
    for waste in current_wastes:
        center = box_center(waste["box"])
        nearest_prev = None
        for prev_center, _prev_class in zone_last_waste_centers:
            dist = calculate_distance(center, prev_center)
            if nearest_prev is None or dist < nearest_prev:
                nearest_prev = dist

        if nearest_prev is None:
            candidate = waste
            candidate_frame_motion = 0.0
            break

        if nearest_prev >= PERSONLESS_MIN_FRAME_MOTION:
            candidate = waste
            candidate_frame_motion = float(nearest_prev)
            break

    if candidate is not None:
        center = box_center(candidate["box"])

        # Start or continue a tracked candidate.
        if zone_candidate_center is None:
            zone_candidate_center = center
            zone_candidate_start_center = center
            zone_candidate_last_center = center
            zone_candidate_cumulative_motion = 0.0
            zone_candidate_streak = 1
            zone_candidate_gap = 0
        else:
            link_distance = calculate_distance(center, zone_candidate_center)
            if link_distance <= DUSTBIN_ZONE_CANDIDATE_MATCH:
                step = calculate_distance(center, zone_candidate_last_center)
                zone_candidate_cumulative_motion += step
                zone_candidate_last_center = center
                zone_candidate_center = center
                zone_candidate_streak += 1
                zone_candidate_gap = 0
            else:
                zone_candidate_center = center
                zone_candidate_start_center = center
                zone_candidate_last_center = center
                zone_candidate_cumulative_motion = 0.0
                zone_candidate_streak = 1
                zone_candidate_gap = 0

        zone_candidate_class = candidate.get("class_name", "WASTE")
        zone_candidate_confidence = float(candidate.get("confidence", 0.0))
        zone_candidate_box = list(candidate["box"])

        displacement = calculate_distance(
            zone_candidate_start_center,
            zone_candidate_last_center
        ) if zone_candidate_start_center and zone_candidate_last_center else 0.0

        current_dumping_reason = (
            "PERSON-INDEPENDENT MOTION: "
            f"{zone_candidate_streak}/{PERSONLESS_CONFIRM_FRAMES} | "
            f"step={candidate_frame_motion:.1f}px | "
            f"path={zone_candidate_cumulative_motion:.1f}px | "
            f"disp={displacement:.1f}px"
        )

        # Confirmation requires BOTH persistence, meaningful trajectory, and downward drop.
        enough_frames = zone_candidate_streak >= PERSONLESS_CONFIRM_FRAMES
        enough_path = zone_candidate_cumulative_motion >= PERSONLESS_MIN_CUMULATIVE_MOTION
        enough_displacement = displacement >= PERSONLESS_MIN_FRAME_MOTION * 1.5

        disp_y = (
            zone_candidate_last_center[1] - zone_candidate_start_center[1]
        ) if zone_candidate_start_center and zone_candidate_last_center else 0.0
        downward_ok = (disp_y >= 25.0) if PERSONLESS_REQUIRE_DOWNWARD_OR_LARGE_MOTION else True

        if enough_frames and enough_path and enough_displacement and downward_ok:
            # Check if candidate fell into the Dustbin
            if is_waste_inside_dustbin(zone_candidate_last_center, frame):
                current_legal_disposal = True
                current_illegal_dumping = False
                last_legal_disposal_time = now
                current_dumping_reason = "LEGAL DISPOSAL: OBJECT LANDED IN DUSTBIN"
                print("🟢 LEGAL DISPOSAL CONFIRMED: Object thrown/landed inside Dustbin!")
                zone_candidate_center = None
                zone_candidate_start_center = None
                zone_candidate_last_center = None
                zone_candidate_streak = 0
                return False

            evidence_waste = {
                "box": list(zone_candidate_box or candidate["box"]),
                "confidence": zone_candidate_confidence,
                "class_name": zone_candidate_class,
                "class_id": candidate.get("class_id", 0),
                "source": "personless_motion"
            }

            evidence_image = create_evidence_image(
                frame,
                [],
                [evidence_waste]
            )

            saved = save_detection(
                waste_type=zone_candidate_class,
                confidence=zone_candidate_confidence,
                source="personless_motion",
                person_detected=False,
                illegal_dumping=True,
                interaction_detected=False,
                evidence_image=evidence_image,
                person_track_id=None
            )

            last_dumping_event_time = now
            last_illegal_dumping_time = now
            zone_last_trigger_time = now
            current_illegal_dumping = True
            current_dumping_reason = (
                "PERSON-INDEPENDENT THROW/DROP CONFIRMED"
            )

            print()
            print("=" * 60)
            print("ILLEGAL DUMPING CONFIRMED - PERSON NOT REQUIRED")
            print("Waste:", zone_candidate_class)
            print("Reason: sustained object trajectory")
            print("Path motion:", round(zone_candidate_cumulative_motion, 1), "pixels")
            print("Displacement:", round(displacement, 1), "pixels")
            print("Evidence:", evidence_image)
            print("Database saved:", saved)
            print("=" * 60)

            zone_entry_streak = 0
            zone_candidate_streak = 0
            zone_candidate_gap = 0
            zone_candidate_center = None
            zone_candidate_start_center = None
            zone_candidate_last_center = None
            zone_candidate_cumulative_motion = 0.0
            zone_last_waste_centers = [
                (box_center(w["box"]), w.get("class_name", "WASTE"))
                for w in current_wastes
            ]
            return True

    else:
        if zone_candidate_center is not None:
            zone_candidate_gap += 1
            if zone_candidate_gap > PERSONLESS_MAX_GAP:
                zone_candidate_center = None
                zone_candidate_streak = 0
                zone_candidate_gap = 0
                zone_candidate_start_center = None
                zone_candidate_last_center = None
                zone_candidate_cumulative_motion = 0.0

        zone_entry_streak = 0
        current_dumping_reason = "PERSON-INDEPENDENT MONITORING - STABLE SCENE"

    zone_last_waste_centers = [
        (box_center(w["box"]), w.get("class_name", "WASTE"))
        for w in current_wastes
    ]

    return False


# ============================================================
# DRAW PERSONS
# ============================================================

def draw_persons(frame, persons):
    for person in persons:
        box = person.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        confidence = float(person.get("confidence", 0.0))
        track_id = person.get("track_id")

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        label = "PERSON"
        if track_id is not None:
            label += f" ID:{track_id}"
        label += f" {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            2
        )

        for wrist in person.get("wrists", []):
            if wrist and len(wrist) == 2:
                wx, wy = int(wrist[0]), int(wrist[1])
                cv2.circle(
                    frame,
                    (wx, wy),
                    5,
                    (255, 0, 255),
                    -1
                )


# ============================================================
# DRAW WASTE
# ============================================================

def draw_wastes(frame, wastes):
    for waste in wastes:
        box = waste.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        class_name = str(waste.get("class_name", "WASTE"))
        confidence = float(waste.get("confidence", 0.0))

        is_inside_dustbin = is_waste_inside_dustbin(waste, frame)
        is_approaching = is_waste_approaching_dustbin(waste, frame)

        if is_inside_dustbin:
            # Legal disposal inside dustbin
            color = (0, 220, 0)
            label = f"LEGAL DUSTBIN: {class_name} {confidence:.2f}"
        elif is_approaching:
            # Reaching to put waste in dustbin -> LEGAL DISPOSAL ACTION!
            color = (220, 180, 0)
            label = f"DISPOSING IN DUSTBIN: {class_name} {confidence:.2f} [LEGAL]"
        else:
            # Outside dustbin on floor -> ILLEGAL!
            color = (0, 0, 255)
            label = f"ILLEGAL WASTE: {class_name} {confidence:.2f} [OUTSIDE DUSTBIN]"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_y = max(18, y1 - 6)
        cv2.rectangle(
            frame,
            (x1, tag_y - th - 3),
            (x1 + tw + 6, tag_y + 3),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )


# ============================================================
# DRAW LIVE STATUS
# ============================================================

def draw_status(frame, persons, wastes):
    height, width = frame.shape[:2]
    now = time.time()
    is_illegal_action = bool(current_illegal_dumping and (now - last_illegal_dumping_time <= ALERT_VISIBLE_SECONDS))
    is_legal_action = bool(current_legal_disposal and (now - last_legal_disposal_time <= LEGAL_ALERT_VISIBLE_SECONDS))

    unheld_illegal_wastes = [
        w for w in (wastes or [])
        if not is_waste_inside_dustbin(w, frame)
        and not is_waste_approaching_dustbin(w, frame)
        and not any(valid_waste_for_holding(p, w) for p in (persons or []))
    ]
    legal_wastes_inside = [w for w in (wastes or []) if is_waste_inside_dustbin(w, frame)]
    approaching_wastes = [w for w in (wastes or []) if is_waste_approaching_dustbin(w, frame)]

    if is_illegal_action:
        status = "STATE: ILLEGAL DUMPING (PERSON DUMPED OUTSIDE DUSTBIN)!"
        background = (0, 0, 220)

    elif is_legal_action or len(approaching_wastes) > 0:
        status = "STATE: LEGAL DISPOSAL (PUTTING WASTE IN DUSTBIN)"
        background = (0, 180, 0)

    elif current_releasing:
        status = "STATE: RELEASING / THROWING WASTE"
        background = (0, 100, 220)

    elif current_holding:
        status = "STATE: PERSON HOLDING WASTE (MONITORING)"
        background = (0, 140, 220)

    elif len(unheld_illegal_wastes) > 0:
        status = f"STATE: ILLEGAL DUMPING ({len(unheld_illegal_wastes)} WASTE OUTSIDE DUSTBIN)!"
        background = (0, 0, 220)

    elif len(legal_wastes_inside) > 0:
        status = f"STATE: MONITORING ({len(legal_wastes_inside)} WASTE IN DUSTBIN)"
        background = (0, 140, 50)

    elif persons:
        status = "STATE: PERSON MONITORING"
        background = (45, 45, 45)

    else:
        status = "STATE: NORMAL (MONITORING)"
        background = (35, 35, 35)

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 38),
        background,
        -1
    )

    cv2.putText(
        frame,
        status,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2
    )


# ============================================================
# GENERIC OBJECT MOTION / THROW FALLBACK
# ============================================================

def _motion_person_mask(shape, persons):
    """Mask detected people so walking limbs/clothing are less likely to be
    mistaken for thrown objects. A small margin accounts for box jitter.
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    scale_x = width / float(CAMERA_WIDTH)
    scale_y = height / float(CAMERA_HEIGHT)

    for person in persons or []:
        box = person.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        pad_x = max(8, int((x2 - x1) * MOTION_PERSON_PAD_X_RATIO))
        pad_y = max(6, int((y2 - y1) * MOTION_PERSON_PAD_Y_RATIO))
        x1 = int(max(0, (x1 - pad_x) * scale_x))
        y1 = int(max(0, (y1 - pad_y) * scale_y))
        x2 = int(min(CAMERA_WIDTH, x2 + pad_x) * scale_x)
        y2 = int(min(CAMERA_HEIGHT, y2 + pad_y) * scale_y)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def detect_motion_blobs(frame, persons):
    """Return moving foreground blobs in full-frame coordinates.

    This supplements YOLO for objects that are not represented in the current
    model's classes. It is trajectory-based, not semantic object recognition.
    """
    global motion_warmup_counter

    small = cv2.resize(
        frame,
        (MOTION_FRAME_WIDTH, MOTION_FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (5, 5), 0)
    foreground = motion_subtractor.apply(small, learningRate=0.003)
    motion_warmup_counter += 1

    if motion_warmup_counter <= MOTION_WARMUP_FRAMES:
        return []

    foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)[1]
    person_mask = _motion_person_mask(foreground.shape, persons)
    foreground[person_mask > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)

    foreground_ratio = cv2.countNonZero(foreground) / float(foreground.size)
    # Broad global changes (camera shake, exposure changes) are not object throws.
    if foreground_ratio > MOTION_MAX_FOREGROUND_RATIO:
        return []

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MOTION_MIN_CONTOUR_AREA or area > MOTION_MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Reject tiny/skinny edge flicker, folds in cloth, and pixel noise.
        # Measurements here are at half resolution, so 12x10 means about
        # 24x20 pixels in the delivered 640x480 frame.
        if w < 12 or h < 10 or w > 210 or h > 180:
            continue

        # Convert half-resolution boxes back to the app's 640x480 coordinates.
        box = [
            max(0, int(x * CAMERA_WIDTH / MOTION_FRAME_WIDTH)),
            max(0, int(y * CAMERA_HEIGHT / MOTION_FRAME_HEIGHT)),
            min(CAMERA_WIDTH, int((x + w) * CAMERA_WIDTH / MOTION_FRAME_WIDTH)),
            min(CAMERA_HEIGHT, int((y + h) * CAMERA_HEIGHT / MOTION_FRAME_HEIGHT))
        ]
        box_w = max(0, box[2] - box[0])
        box_h = max(0, box[3] - box[1])
        # An incident candidate must occupy a meaningful image area. Smaller
        # details remain visible through the YOLO waste/bottle detector path.
        if box_w * box_h < 800:
            continue
        center = box_center(box)

        # Double-check that the blob center is outside expanded person boxes.
        center_inside_person = False
        for person in persons or []:
            pbox = person.get("box")
            if not pbox or len(pbox) != 4:
                continue
            px1, py1, px2, py2 = [int(v) for v in pbox]
            pad_x = max(8, int((px2 - px1) * MOTION_PERSON_PAD_X_RATIO))
            pad_y = max(6, int((py2 - py1) * MOTION_PERSON_PAD_Y_RATIO))
            if px1 - pad_x <= center[0] <= px2 + pad_x and py1 - pad_y <= center[1] <= py2 + pad_y:
                center_inside_person = True
                break
        if center_inside_person:
            continue

        # A blob mostly contained inside a person box is likely a moving body part.
        if any(intersection_over_waste(box, p.get("box", [0, 0, 0, 0])) >= 0.45
               for p in (persons or []) if p.get("box")):
            continue

        blobs.append({
            "box": box,
            "center": center,
            "area": area * (CAMERA_WIDTH / MOTION_FRAME_WIDTH) * (CAMERA_HEIGHT / MOTION_FRAME_HEIGHT),
            "confidence": 0.50,
            "class_name": "MOVING OBJECT",
            "source": "motion_fallback",
            "semantic_match": False
        })

    # Larger clean blobs are preferred when starting a new trajectory.
    blobs.sort(key=lambda blob: blob["area"], reverse=True)
    return blobs[:12]


def _match_motion_blob_to_detection(blob, wastes):
    """Use a nearby YOLO waste/object box to label the motion trajectory if available."""
    if not wastes:
        return blob
    cx, cy = blob["center"]
    best = None
    best_distance = None
    for det in wastes:
        box = det.get("box")
        if not box or len(box) != 4:
            continue
        dist = calculate_distance((cx, cy), box_center(box))
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best = det
    if best is not None and best_distance is not None and best_distance <= 100:
        blob = dict(blob)
        blob["class_name"] = str(best.get("class_name", "MOVING OBJECT"))
        blob["confidence"] = float(best.get("confidence", 0.50))
        blob["class_id"] = best.get("class_id", 0)
        blob["semantic_match"] = True
    return blob


def process_motion_fallback(frame, persons, wastes):
    """Confirm a thrown/dropped object from its motion path even if YOLO cannot
    identify its category. This runs with or without a visible person.
    """
    global motion_candidate_center, motion_candidate_start_center
    global motion_candidate_last_center, motion_candidate_box
    global motion_candidate_streak, motion_candidate_gap
    global motion_candidate_cumulative_motion, motion_candidate_label
    global motion_candidate_confidence, current_motion_box, current_motion_label
    global last_motion_event_time, last_dumping_event_time, last_illegal_dumping_time
    global current_illegal_dumping, current_dumping_reason, zone_last_trigger_time

    if not ENABLE_MOTION_FALLBACK_DUMPING:
        # Generic pixel motion without verified waste model detection
        # is disabled to eliminate false positives.
        return False

    now = time.time()
    blobs = detect_motion_blobs(frame, persons)
    current_motion_box = None
    current_motion_label = ""

    chosen = None
    if blobs:
        if motion_candidate_center is None:
            chosen = blobs[0]
        else:
            nearest = min(
                blobs,
                key=lambda blob: calculate_distance(blob["center"], motion_candidate_center)
            )
            nearest_dist = calculate_distance(nearest["center"], motion_candidate_center)
            if nearest_dist <= MOTION_TRACK_MAX_STEP:
                chosen = nearest
            elif motion_candidate_gap > 0:
                # A new trajectory can begin after an interrupted/ended candidate.
                chosen = blobs[0]

    if chosen is None:
        if motion_candidate_center is not None:
            motion_candidate_gap += 1
            if motion_candidate_gap > MOTION_MAX_GAP:
                motion_candidate_center = None
                motion_candidate_start_center = None
                motion_candidate_last_center = None
                motion_candidate_box = None
                motion_candidate_streak = 0
                motion_candidate_gap = 0
                motion_candidate_cumulative_motion = 0.0
        return False

    chosen = _match_motion_blob_to_detection(chosen, wastes)
    center = chosen["center"]

    if motion_candidate_center is None or motion_candidate_gap > MOTION_MAX_GAP:
        motion_candidate_center = center
        motion_candidate_start_center = center
        motion_candidate_last_center = center
        motion_candidate_box = list(chosen["box"])
        motion_candidate_streak = 1
        motion_candidate_gap = 0
        motion_candidate_cumulative_motion = 0.0
    else:
        step = calculate_distance(center, motion_candidate_last_center)
        if step > MOTION_TRACK_MAX_STEP:
            motion_candidate_center = center
            motion_candidate_start_center = center
            motion_candidate_last_center = center
            motion_candidate_box = list(chosen["box"])
            motion_candidate_streak = 1
            motion_candidate_gap = 0
            motion_candidate_cumulative_motion = 0.0
        else:
            motion_candidate_cumulative_motion += step
            motion_candidate_last_center = center
            motion_candidate_center = center
            motion_candidate_box = list(chosen["box"])
            motion_candidate_streak += 1
            motion_candidate_gap = 0

    motion_candidate_label = str(chosen.get("class_name", "MOVING OBJECT"))
    motion_candidate_confidence = float(chosen.get("confidence", 0.50))
    current_motion_box = list(motion_candidate_box) if motion_candidate_box else None
    current_motion_label = motion_candidate_label

    displacement = calculate_distance(
        motion_candidate_start_center,
        motion_candidate_last_center
    ) if motion_candidate_start_center and motion_candidate_last_center else 0.0

    # Recognized waste detections can use a slightly lower trajectory threshold;
    # unknown blobs need stronger evidence to avoid treating fabric edges,
    # lighting changes, and detector jitter as illegal dumping.
    semantic_match = bool(chosen.get("semantic_match", False))
    if semantic_match:
        normal_trajectory = (
            motion_candidate_streak >= 4
            and motion_candidate_cumulative_motion >= 90.0
            and displacement >= 50.0
        )
        fast_trajectory = (
            motion_candidate_streak >= 4
            and motion_candidate_cumulative_motion >= 180.0
            and displacement >= 110.0
        )
    else:
        normal_trajectory = (
            motion_candidate_streak >= MOTION_MIN_TRACK_FRAMES
            and motion_candidate_cumulative_motion >= MOTION_MIN_PATH_PIXELS
            and displacement >= MOTION_MIN_DISPLACEMENT_PIXELS
        )
        fast_trajectory = (
            motion_candidate_streak >= 4
            and motion_candidate_cumulative_motion >= MOTION_FAST_PATH_PIXELS
            and displacement >= MOTION_FAST_DISPLACEMENT_PIXELS
        )

    # Prevent duplicate motion events and suppress repeats of the same visual
    # disturbance. Known-waste release logic still has its own confirmation path.
    duplicate_recent = now - last_illegal_dumping_time < 8.0
    if not (normal_trajectory or fast_trajectory) or duplicate_recent:
        return False
    if now - last_motion_event_time < MOTION_EVENT_COOLDOWN:
        return False

    evidence_object = {
        "box": list(motion_candidate_box or chosen["box"]),
        "confidence": motion_candidate_confidence,
        "class_name": motion_candidate_label or "MOVING OBJECT",
        "class_id": chosen.get("class_id", 0),
        "source": "motion_fallback"
    }
    evidence_image = create_evidence_image(frame, persons or [], [evidence_object])

    nearest_person_id = None
    if persons:
        nearest_person = min(
            persons,
            key=lambda person: calculate_distance(
                box_center(person.get("box", [0, 0, 0, 0])),
                center
            )
        )
        nearest_person_id = nearest_person.get("track_id")

    saved = save_detection(
        waste_type=evidence_object["class_name"],
        confidence=evidence_object["confidence"],
        source="motion_fallback",
        person_detected=bool(persons),
        illegal_dumping=True,
        interaction_detected=bool(persons),
        evidence_image=evidence_image,
        person_track_id=nearest_person_id
    )

    last_motion_event_time = now
    last_dumping_event_time = now
    last_illegal_dumping_time = now
    zone_last_trigger_time = now
    current_illegal_dumping = True
    current_dumping_reason = (
        f"MOTION THROW/DROP CONFIRMED: {evidence_object['class_name']} | "
        f"path={motion_candidate_cumulative_motion:.1f}px | disp={displacement:.1f}px"
    )

    print("\n" + "=" * 60)
    print("ILLEGAL DUMPING CONFIRMED - MOTION FALLBACK")
    print("Object:", evidence_object["class_name"])
    print("Persons visible:", bool(persons))
    print("Trajectory frames:", motion_candidate_streak)
    print("Path motion:", round(motion_candidate_cumulative_motion, 1), "pixels")
    print("Displacement:", round(displacement, 1), "pixels")
    print("Evidence:", evidence_image)
    print("Database saved:", saved)
    print("=" * 60 + "\n")

    # Reset this trajectory so a single throw cannot be counted repeatedly.
    motion_candidate_center = None
    motion_candidate_start_center = None
    motion_candidate_last_center = None
    motion_candidate_box = None
    motion_candidate_streak = 0
    motion_candidate_gap = 0
    motion_candidate_cumulative_motion = 0.0
    current_motion_box = None
    current_motion_label = ""
    return True


def draw_motion_candidate(frame):
    """Show a yellow tracking box while a generic moving-object candidate forms."""
    if not current_motion_box:
        return
    x1, y1, x2, y2 = [int(v) for v in current_motion_box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    label = f"MOTION CANDIDATE: {current_motion_label or 'OBJECT'}"
    cv2.putText(
        frame,
        label,
        (x1, max(42, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 255),
        2
    )


# ============================================================
# DRAW DUSTBIN / DUMPING ZONE
# ============================================================

def draw_dustbin_zone(frame):
    """Draw clean, tight outline of the detected physical dustbin receptacle."""
    if not DUSTBIN_ZONE_ENABLED:
        return
    db = get_effective_dustbin_zone(frame)
    if db is None:
        return
    dx1, dy1, dx2, dy2 = [int(v) for v in db]
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 220, 0), 2)
    label = "DUSTBIN (LEGAL RECEPTACLE)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tag_y = max(20, dy1 - 5)
    cv2.rectangle(frame, (dx1, tag_y - th - 3), (dx1 + tw + 6, tag_y + 3), (0, 180, 0), -1)
    cv2.putText(frame, label, (dx1 + 3, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


class PhoneMJPEGCapture:
    """Reliable MJPEG reader for IP Webcam Pro / MJPEG HTTP streams.

    OpenCV's cv2.VideoCapture may report the HTTP stream as opened while
    still returning no decoded frames. This reader pulls the multipart byte
    stream directly, extracts JPEG SOI/EOI markers, and decodes each image.
    """

    def __init__(self, url, timeout=(5, 15)):
        self.url = url
        self.timeout = timeout
        self.response = None
        self.iterator = None
        self.buffer = bytearray()
        self.opened = False
        self._connect()

    def _connect(self):
        self.release()
        print("Opening PHONE MJPEG stream:", self.url)
        response = requests.get(
            self.url,
            stream=True,
            timeout=self.timeout,
            headers={"User-Agent": "AI-Waste-Detection/1.0"}
        )
        response.raise_for_status()
        self.response = response
        self.iterator = response.iter_content(chunk_size=4096)
        self.buffer.clear()
        self.opened = True

    def isOpened(self):
        return bool(self.opened and self.response is not None)

    def set(self, *_args, **_kwargs):
        # Compatibility with cv2.VideoCapture.set().
        return True

    def release(self):
        self.opened = False
        self.buffer.clear()
        iterator = self.iterator
        self.iterator = None
        if iterator is not None:
            try:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
        self.response = None

    def read(self):
        if not self.isOpened():
            return False, None

        reconnect_attempted = False

        while self.isOpened():
            start = self.buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    jpeg_bytes = bytes(self.buffer[start:end + 2])
                    del self.buffer[:end + 2]

                    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
                    continue

                # Throw away bytes before the JPEG start marker to prevent
                # unbounded buffer growth from multipart headers.
                if start > 0:
                    del self.buffer[:start]

            try:
                chunk = next(self.iterator)
            except StopIteration:
                if not reconnect_attempted:
                    reconnect_attempted = True
                    try:
                        self._connect()
                        continue
                    except Exception as exc:
                        print("Phone camera reconnect error:", exc)
                        return False, None
                return False, None
            except Exception as exc:
                if not reconnect_attempted:
                    reconnect_attempted = True
                    print("Phone camera stream read error:", exc)
                    try:
                        self._connect()
                        continue
                    except Exception as reconnect_exc:
                        print("Phone camera reconnect error:", reconnect_exc)
                        return False, None
                return False, None

            if chunk:
                self.buffer.extend(chunk)

            # Keep the parser bounded if the stream sends non-JPEG data.
            if len(self.buffer) > 4 * 1024 * 1024:
                marker = self.buffer.find(b"\xff\xd8")
                if marker >= 0:
                    del self.buffer[:marker]
                else:
                    self.buffer.clear()

        return False, None


def get_camera():

    global camera

    with camera_lock:

        if camera is None:

            print(
                "Opening Camera..."
            )

            if CAMERA_MODE.upper() == "PHONE":
                if not PHONE_CAMERA_URL or not PHONE_CAMERA_URL.startswith(("http://", "https://")):
                    print("ERROR: Set PHONE_CAMERA_URL to your phone camera stream URL.")
                    camera = None
                    return None

                try:
                    camera = PhoneMJPEGCapture(PHONE_CAMERA_URL)
                except Exception as exc:
                    print("ERROR: Phone camera stream could not be opened:", exc)
                    camera = None
                    return None

            else:
                print("Opening PC camera...")
                camera = cv2.VideoCapture(
                    0,
                    cv2.CAP_DSHOW
                )

                # Some laptop webcams behave better without forced
                # DirectShow/MJPG settings. If DirectShow fails, fall back
                # to OpenCV's default backend.
                if camera is None or not camera.isOpened():
                    try:
                        if camera is not None:
                            camera.release()
                    except Exception:
                        pass
                    print("DirectShow camera open failed. Trying default camera backend...")
                    camera = cv2.VideoCapture(0)

                if camera is not None:
                    try:
                        camera.set(
                            cv2.CAP_PROP_BUFFERSIZE,
                            CAMERA_BUFFER_SIZE
                        )
                    except Exception:
                        pass

                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    CAMERA_WIDTH
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    CAMERA_HEIGHT
                )

                camera.set(
                    cv2.CAP_PROP_FPS,
                    CAMERA_FPS
                )

            if camera is None or not camera.isOpened():

                print(
                    "ERROR: Camera could not be opened."
                )

                try:
                    if camera is not None:
                        camera.release()
                except Exception:
                    pass

                camera = None

                return None

            print(
                "Camera Opened Successfully"
            )

    return camera


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    global camera
    global current_person_count
    global current_waste_count

    global current_tracked_person_ids

    global current_illegal_dumping
    global current_legal_disposal

    global last_illegal_dumping_time
    global last_illegal_dumping_save_time
    global last_legal_disposal_time
    global last_evidence_image

    global last_person_save_time
    global last_waste_save_time
    global last_interaction_save_time
    global personless_empty_streak
    global latest_frame_bytes
    global latest_frame_time
    global stream_error
    global stream_running
    global cached_persons
    global cached_wastes
    global recorded_waste_centers
    global recorded_illegal_dumps

    stream_running = True
    stream_error = ""

    cap = get_camera()

    if cap is None:
        stream_running = False
        stream_error = "Camera could not be opened"
        return

    frame_index = 0
    if not isinstance(cached_persons, list):
        cached_persons = []
    else:
        cached_persons.clear()

    if not isinstance(cached_wastes, list):
        cached_wastes = []
    else:
        cached_wastes.clear()
    consecutive_read_failures = 0

    while True:

        if stream_stop_event.is_set():
            break

        success, frame = cap.read()

        if not success:
            consecutive_read_failures += 1
            print(
                f"WARNING: Camera read failed "
                f"({consecutive_read_failures}/{CAMERA_READ_FAILURE_LIMIT})"
            )
            time.sleep(0.05)

            if consecutive_read_failures >= CAMERA_READ_FAILURE_LIMIT:
                stream_error = "Camera frame read failed; reconnecting"
                print("Camera stream lost after repeated read failures. Reopening camera...")
                with camera_lock:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    camera = None

                time.sleep(CAMERA_RECONNECT_DELAY)
                cap = get_camera()
                consecutive_read_failures = 0

                if cap is None:
                    # Do not terminate the generator when the webcam is
                    # temporarily unavailable. Keep retrying instead.
                    time.sleep(CAMERA_RECONNECT_DELAY)

                continue

            continue

        consecutive_read_failures = 0
        stream_error = ""

        frame = cv2.resize(
            frame,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

        frame_index += 1

        # ====================================================
        # DETECTION
        # Run the expensive AI inference only every Nth frame.
        # Cached detections are reused in between to keep the video
        # stream smooth and reduce CPU/GPU load.
        # ====================================================

        run_inference = (frame_index % INFERENCE_EVERY_N_FRAMES == 1)

        if run_inference:

            cached_persons = detect_persons(
                frame
            ) or []

            detected = detect_waste(
                frame
            ) or []

            cached_wastes = filter_false_waste_detections(
                detected,
                cached_persons,
                frame
            ) or []

            print(
                f"[DETECTION] persons={len(cached_persons)} "
                f"wastes={len(cached_wastes)} "
                f"classes={[w.get('class_name') for w in cached_wastes]}"
            )

        persons = cached_persons
        wastes = cached_wastes


        # ====================================================
        # LIVE COUNTERS
        # ====================================================

        current_person_count = len(
            persons
        )

        current_waste_count = len(
            wastes
        )

        current_tracked_person_ids = [

            person.get(
                "track_id"
            )

            for person in persons

            if person.get(
                "track_id"
            ) is not None
        ]


        # ====================================================
        # DUMPING PROCESS
        # ====================================================

        dumping_confirmed = False

        if run_inference:
            # Person throw/drop detection: ONLY person throwing waste outside dustbin triggers dumping!
            dumping_confirmed = process_dumping(
                persons,
                wastes,
                frame
            )

        now = time.time()

        # Check wastes outside vs inside dustbin for real-time counts
        unheld_illegal_wastes = [
            w for w in wastes
            if not is_waste_inside_dustbin(w, frame)
            and not is_waste_approaching_dustbin(w, frame)
            and not any(valid_waste_for_holding(p, w) for p in persons)
        ]
        legal_wastes_inside = [w for w in wastes if is_waste_inside_dustbin(w, frame)]
        approaching_wastes = [w for w in wastes if is_waste_approaching_dustbin(w, frame)]

        if approaching_wastes:
            current_legal_disposal = True
            last_legal_disposal_time = now

        if dumping_confirmed:
            current_illegal_dumping = True
            last_illegal_dumping_time = now
            print(f"🚨 [PERSON ILLEGAL DUMPING ACTION] Person threw waste outside dustbin!")
        elif unheld_illegal_wastes:
            current_dumping_reason = f"ILLEGAL DUMPING: {len(unheld_illegal_wastes)} waste outside dustbin"
            # Auto-save evidence of illegal waste lying outside dustbin
            if now - last_illegal_dumping_save_time >= ILLEGAL_DUMPING_SAVE_COOLDOWN:
                save_needed = False
                for w in unheld_illegal_wastes:
                    wc = box_center(w["box"])
                    grid_key = (int(wc[0] // 50), int(wc[1] // 50))
                    if now - recorded_illegal_dumps.get(grid_key, 0) >= 30.0:
                        save_needed = True
                        recorded_illegal_dumps[grid_key] = now

                if save_needed:
                    evidence_img = create_evidence_image(frame, persons, unheld_illegal_wastes)
                    for w in unheld_illegal_wastes:
                        nearest_p = None
                        if persons:
                            nearest_p = min(persons, key=lambda p: calculate_distance(box_center(p["box"]), box_center(w["box"])))
                        p_id = nearest_p.get("track_id") if nearest_p else None
                        save_detection(
                            waste_type=w.get("class_name", "WASTE"),
                            confidence=float(w.get("confidence", 0.0)),
                            source="camera",
                            person_detected=bool(persons),
                            illegal_dumping=True,
                            interaction_detected=False,
                            evidence_image=evidence_img,
                            person_track_id=p_id
                        )
                    last_illegal_dumping_save_time = now
                    last_illegal_dumping_time = now
                    print(f"🚨 [ILLEGAL DUMPING RECORDED] {len(unheld_illegal_wastes)} waste item(s) outside dustbin!")

        # Reset alert action flag after visible cooldown
        if now - last_illegal_dumping_time > ALERT_VISIBLE_SECONDS:
            current_illegal_dumping = False

        # Auto-reset legal disposal banner after LEGAL_ALERT_VISIBLE_SECONDS
        if now - last_legal_disposal_time > LEGAL_ALERT_VISIBLE_SECONDS:
            current_legal_disposal = False


        # ====================================================
        # NORMAL DATABASE SAVING
        #
        # PERSON ONLY -> SAVE PERSON DETECTION
        # WASTE       -> SAVE WASTE DETECTION
        # ====================================================

        now = time.time()


        # ----------------------------------------------------
        # PERSON SAVE
        # ----------------------------------------------------

        if (
            persons
            and
            now - last_person_save_time
            >= NORMAL_SAVE_INTERVAL
        ):

            for person in persons:

                save_detection(

                    waste_type=None,

                    confidence=
                        person["confidence"],

                    source="camera",

                    person_detected=True,

                    illegal_dumping=False,

                    interaction_detected=False,

                    evidence_image=None,

                    person_track_id=
                        person.get("track_id")
                )


            last_person_save_time = now


        # ----------------------------------------------------
        # WASTE SAVE
        # ----------------------------------------------------

        if (
            wastes
            and
            now - last_waste_save_time
            >= NORMAL_SAVE_INTERVAL
        ):

            for waste in wastes:

                person_id = None

                if persons:

                    nearest_person = min(

                        persons,

                        key=lambda p:
                        calculate_distance(

                            box_center(
                                p["box"]
                            ),

                            box_center(
                                waste["box"]
                            )
                        )
                    )

                    person_id = (
                        nearest_person.get(
                            "track_id"
                        )
                    )


                save_detection(

                    waste_type=
                        waste["class_name"],

                    confidence=
                        waste["confidence"],

                    source="camera",

                    person_detected=
                        bool(persons),

                    illegal_dumping=
                        False,

                    interaction_detected=
                        current_holding,

                    evidence_image=None,

                    person_track_id=
                        person_id
                )


            last_waste_save_time = now


        # ----------------------------------------------------
        # INTERACTION SAVE
        # ----------------------------------------------------

        if (
            current_holding
            and
            now - last_interaction_save_time
            >= NORMAL_SAVE_INTERVAL
        ):

            interaction_person_id = None

            if holding_person_id is not None:

                interaction_person_id = (
                    holding_person_id
                )

            save_detection(

                waste_type="INTERACTION",

                confidence=1.0,

                source="camera",

                person_detected=True,

                illegal_dumping=False,

                interaction_detected=True,

                evidence_image=None,

                person_track_id=
                    interaction_person_id
            )


            last_interaction_save_time = now


        # ====================================================
        # DRAW
        # ====================================================

        draw_persons(
            frame,
            persons
        )

        draw_wastes(
            frame,
            wastes
        )

        draw_motion_candidate(frame)

        draw_dustbin_zone(
            frame
        )

        draw_status(
            frame,
            persons,
            wastes
        )


        # ====================================================
        # ENCODE
        # ====================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )


        if not ret:
            continue


        frame_bytes = buffer.tobytes()

        with stream_lock:
            latest_frame_bytes = frame_bytes
            latest_frame_time = time.time()

        yield frame_bytes


    # Keep the global camera open here. /stop_camera owns the camera lifecycle.


# ============================================================
# BACKGROUND CAMERA STREAM WORKER
# ============================================================

def ensure_stream_worker():

    global stream_thread

    with stream_lock:
        if stream_thread is not None and stream_thread.is_alive():
            return

        stream_stop_event.clear()
        stream_thread = threading.Thread(
            target=camera_stream_worker,
            name="camera-stream-worker",
            daemon=True
        )
        stream_thread.start()


def camera_stream_worker():

    global stream_thread
    global stream_running
    global stream_error

    try:
        for _frame_bytes in generate_frames():
            if stream_stop_event.is_set():
                break

    except Exception as exc:
        stream_error = f"{type(exc).__name__}: {exc}"
        print("Camera stream worker error:", stream_error)

    finally:
        stream_running = False
        stream_thread = None


def cached_mjpeg_stream():

    ensure_stream_worker()

    while True:
        if stream_stop_event.is_set():
            break

        with stream_lock:
            frame_bytes = latest_frame_bytes

        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame_bytes)}\r\n".encode("ascii")
                + b"Cache-Control: no-cache\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        else:
            time.sleep(0.05)

        time.sleep(0.03)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# CAMERA
# ============================================================

@app.route("/camera")
def camera_page():

    return render_template(
        "camera.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    ensure_stream_worker()

    response = Response(
        cached_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/camera_snapshot")
def camera_snapshot():

    ensure_stream_worker()

    with stream_lock:
        frame_bytes = latest_frame_bytes

    if not frame_bytes:
        return ("Camera frame not ready", 503)

    response = Response(frame_bytes, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    ensure_stream_worker()

    camera_open = bool(
        stream_running
        or latest_frame_bytes is not None
        or (camera is not None and getattr(camera, "isOpened", lambda: False)())
    )

    total_illegal_incidents = 0
    try:
        conn = get_db()
        row = conn.execute("SELECT COUNT(*) FROM detections WHERE illegal_dumping = 1").fetchone()
        if row:
            total_illegal_incidents = int(row[0])
        conn.close()
    except Exception:
        pass

    w_list = cached_wastes if isinstance(cached_wastes, list) else []
    p_list = cached_persons if isinstance(cached_persons, list) else []
    unheld_illegal_wastes = [
        w for w in w_list
        if not is_waste_inside_dustbin(w)
        and not is_waste_approaching_dustbin(w)
        and not any(valid_waste_for_holding(p, w) for p in p_list)
    ]
    curr_illegal_count = len(unheld_illegal_wastes)
    curr_legal_count = len([w for w in w_list if is_waste_inside_dustbin(w)])
    has_approaching = any(is_waste_approaching_dustbin(w) for w in w_list)
    now = time.time()
    is_illegal_action = bool(current_illegal_dumping and (now - last_illegal_dumping_time <= ALERT_VISIBLE_SECONDS))
    is_illegal_active = bool(is_illegal_action or curr_illegal_count > 0)
    is_legal_active = bool(
        current_legal_disposal
        or has_approaching
        or (curr_legal_count > 0 and curr_illegal_count == 0)
        or (now - last_legal_disposal_time <= LEGAL_ALERT_VISIBLE_SECONDS)
    )

    return jsonify(
        {
            "success": True,

            "camera_id":
                CAMERA_ID,

            "location":
                CAMERA_LOCATION,

            "camera_open":
                camera_open,

            "stream_running":
                stream_running,

            "stream_error":
                stream_error,

            "frame_ready":
                latest_frame_bytes is not None,

            "camera_mode":
                CAMERA_MODE,

            "person_count":
                len(p_list),

            "waste_count":
                len(w_list),

            "total_illegal_incidents":
                total_illegal_incidents,

            "interaction":
                current_interaction,

            "holding":
                current_holding,

            "releasing":
                current_releasing,

            "illegal_dumping":
                is_illegal_active,

            "legal_disposal":
                is_legal_active,

            "illegal_waste_count":
                curr_illegal_count,

            "legal_waste_count":
                curr_legal_count,

            "dustbin_zone_enabled":
                DUSTBIN_ZONE_ENABLED,

            "dustbin_zone":
                get_effective_dustbin_zone(),

            "tracked_person_ids":
                current_tracked_person_ids,

            "tracked_person_count":
                len(
                    current_tracked_person_ids
                ),

            "waste_model":
                "ONLINE",

            "person_model":
                "ONLINE",

            "object_tracking":
                "ONLINE",

            "alert_engine":
                "ONLINE",

            "evidence_image":
                last_evidence_image,
            "waste_classes": list(get_custom_model_classes().values()),
            "multi_class_waste_ready": True
        }
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify(

        {
            "success": True,

            "status":
                "healthy",

            "waste_model":
                os.path.exists(
                    WASTE_MODEL_PATH
                ),

            "pose_model":
                os.path.exists(
                    POSE_MODEL_PATH
                ),

            "bottle_model":
                os.path.exists(
                    BOTTLE_MODEL_PATH
                ),

            "camera":
                camera is not None
                and
                camera.isOpened(),

            "database":
                os.path.exists(
                    DATABASE
                )
        }
    )


# ============================================================
# LATEST ALERT
# ============================================================

@app.route("/api/latest-alert")
def latest_alert():
    now = time.time()
    w_list = cached_wastes if isinstance(cached_wastes, list) else []
    p_list = cached_persons if isinstance(cached_persons, list) else []
    unheld_illegal_wastes = [
        w for w in w_list
        if not is_waste_inside_dustbin(w)
        and not is_waste_approaching_dustbin(w)
        and not any(valid_waste_for_holding(p, w) for p in p_list)
    ]
    curr_illegal_count = len(unheld_illegal_wastes)
    is_illegal_action = bool(current_illegal_dumping and (now - last_illegal_dumping_time <= ALERT_VISIBLE_SECONDS))
    is_active = bool(is_illegal_action or curr_illegal_count > 0)

    if not is_active:
        return jsonify(
            {
                "success": True,
                "alert": None
            }
        )

    conn = get_db()
    row = conn.execute(
        """
        SELECT *
        FROM detections
        WHERE illegal_dumping = 1
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    conn.close()

    if row is None:
        return jsonify(
            {
                "success": True,
                "alert": None
            }
        )

    alert = dict(row)

    try:
        detected_time = datetime.strptime(
            alert["detected_at"],
            "%Y-%m-%d %H:%M:%S"
        )
        age = (datetime.now() - detected_time).total_seconds()
        if age > ALERT_VISIBLE_SECONDS and not current_illegal_dumping and curr_illegal_count == 0:
            return jsonify(
                {
                    "success": True,
                    "alert": None
                }
            )
    except Exception:
        pass


    return jsonify(

        {
            "success": True,

            "alert": {

                "id":
                    alert["id"],

                "waste_type":
                    alert["waste_type"],

                "confidence":
                    alert["confidence"],

                "detected_at":
                    alert["detected_at"],

                "camera_id":
                    alert["camera_id"],

                "location":
                    alert["location"],

                "person_track_id":
                    alert["person_track_id"],

                "evidence_image":
                    alert["evidence_image"]
            }
        }
    )


# ============================================================
# ALERTS
# ============================================================

@app.route("/api/alerts")
def api_alerts():

    conn = get_db()


    rows = conn.execute(
        """
        SELECT *

        FROM detections

        WHERE illegal_dumping = 1

        ORDER BY id DESC

        LIMIT 50
        """
    ).fetchall()


    conn.close()


    return jsonify(

        {
            "success": True,

            "alerts": [
                dict(row)
                for row in rows
            ]
        }
    )


# ============================================================
# EVIDENCE IMAGE
# ============================================================

@app.route("/evidence/<path:filename>")
def evidence_image(filename):
    """
    Serve saved illegal-dumping evidence images safely.
    Accepts only the filename/path under static/results.
    """
    safe_name = os.path.basename(filename)

    if not safe_name:
        return "Evidence image not found", 404

    # Primary evidence location.
    filepath = os.path.join(RESULT_FOLDER, safe_name)

    if not os.path.isfile(filepath):
        print("Evidence file not found:", filepath)
        return (
            "Evidence image not found. The database record exists, "
            "but the image file is missing from static/results.",
            404
        )

    return send_from_directory(
        RESULT_FOLDER,
        safe_name
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM detections
        ORDER BY id DESC
        """
    ).fetchall()

    total_detections = len(rows)
    person_detections = sum(1 for r in rows if r["person_detected"])
    interactions = sum(1 for r in rows if r["interaction_detected"])
    illegal_dumping = sum(1 for r in rows if r["illegal_dumping"])

    conn.close()

    return render_template(
        "history.html",
        detections=rows,
        total_detections=total_detections,
        person_detections=person_detections,
        interactions=interactions,
        illegal_dumping=illegal_dumping
    )


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history")
def api_history():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM detections
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return jsonify(
        {
            "success": True,
            "detections": [
                dict(row)
                for row in rows
            ]
        }
    )


# ============================================================
# EXPORT HISTORY CSV
# ============================================================

@app.route("/export_history_csv")
def export_history_csv():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM detections
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow([
        "ID",
        "Waste Type",
        "Confidence",
        "Source",
        "Detected At",
        "Person Detected",
        "Illegal Dumping",
        "Interaction Detected",
        "Camera ID",
        "Location",
        "Evidence Image",
        "Person Track ID"
    ])

    for r in rows:
        writer.writerow([
            r["id"],
            r["waste_type"] or "N/A",
            f"{float(r['confidence'] or 0.0):.2f}",
            r["source"] or "",
            r["detected_at"] or "",
            "YES" if r["person_detected"] else "NO",
            "YES" if r["illegal_dumping"] else "NO",
            "YES" if r["interaction_detected"] else "NO",
            r["camera_id"] or "",
            r["location"] or "",
            r["evidence_image"] or "",
            r["person_track_id"] if r["person_track_id"] is not None else ""
        ])

    output = io.BytesIO(si.getvalue().encode("utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        output,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=waste_detection_history_{timestamp}.csv"
        }
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/clear_history",
    methods=["POST"]
)
def clear_history():

    conn = get_db()


    conn.execute(
        "DELETE FROM detections"
    )


    conn.commit()

    conn.close()


    return jsonify(

        {
            "success": True,

            "message":
                "History cleared"
        }
    )


# ============================================================
# STOP CAMERA
# ============================================================

@app.route(
    "/stop_camera",
    methods=["GET", "POST"]
)
def stop_camera():

    global camera
    global stream_thread

    stream_stop_event.set()

    global current_person_count
    global current_waste_count

    global current_interaction
    global current_holding
    global current_releasing
    global current_illegal_dumping

    global current_tracked_person_ids


    with camera_lock:

        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

            camera = None


    current_person_count = 0
    current_waste_count = 0

    current_interaction = False
    current_holding = False
    current_releasing = False
    current_illegal_dumping = False

    current_tracked_person_ids = []


    reset_dumping_state()


    return jsonify(

        {
            "success": True,

            "message":
                "Camera stopped"
        }
    )


# ============================================================
# IMAGE UPLOAD DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect_image():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image file provided"
        }), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "message": "No file selected"
        }), 400

    raw_name = secure_filename(file.filename) or "upload.jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_filename = f"upload_{timestamp}_{raw_name}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)

    file.save(input_path)

    frame = cv2.imread(input_path)
    if frame is None:
        return jsonify({
            "success": False,
            "message": "Failed to decode image"
        }), 400

    # Run AI Detection
    wastes = detect_waste(frame)
    persons = detect_persons(frame)
    wastes = filter_false_waste_detections(wastes, persons, frame)

    annotated = frame.copy()
    draw_persons(annotated, persons)
    draw_wastes(annotated, wastes)

    interaction = False
    for p in persons:
        for w in wastes:
            if waste_near_hand(p, w, threshold=HAND_WASTE_DISTANCE):
                interaction = True
                break
        if interaction:
            break

    # Add status header
    h, w = annotated.shape[:2]
    header_color = (0, 0, 180) if interaction else ((0, 140, 0) if wastes else (60, 60, 60))
    cv2.rectangle(annotated, (0, 0), (w, 50), header_color, -1)
    status_label = f"WASTE: {len(wastes)} | PERSONS: {len(persons)}" + (" | INTERACTION DETECTED" if interaction else "")
    cv2.putText(
        annotated,
        status_label,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    result_filename = f"detected_{timestamp}_{raw_name}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated)

    # Save to database
    for waste in wastes:
        save_detection(
            waste_type=waste["class_name"],
            confidence=waste["confidence"],
            source="upload",
            person_detected=bool(persons),
            illegal_dumping=False,
            interaction_detected=interaction,
            evidence_image=f"/evidence/{result_filename}"
        )

    return jsonify({
        "success": True,
        "waste_count": len(wastes),
        "person_count": len(persons),
        "interaction_detected": interaction,
        "wastes": wastes,
        "result_image": f"/evidence/{result_filename}",
        "input_image": f"/static/uploads/{input_filename}"
    })


# ============================================================
# CAMERA SETTINGS API
# ============================================================

@app.route("/api/camera_settings", methods=["GET", "POST"])
def camera_settings():

    global CAMERA_MODE, PHONE_CAMERA_URL, camera
    global DUSTBIN_ZONE_ENABLED, DUSTBIN_ZONE

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        new_mode = str(data.get("camera_mode", "")).strip().upper()
        new_url = str(data.get("phone_url", "")).strip()

        camera_reconnect_needed = False
        if new_mode in ("PC", "PHONE") and new_mode != CAMERA_MODE:
            CAMERA_MODE = new_mode
            camera_reconnect_needed = True

        if new_url and new_url != PHONE_CAMERA_URL:
            PHONE_CAMERA_URL = new_url
            camera_reconnect_needed = True

        # Dustbin zone settings
        if "dustbin_enabled" in data:
            val = data.get("dustbin_enabled")
            DUSTBIN_ZONE_ENABLED = True if val in (True, "true", "True", "1", 1) else False

        if "dustbin_position" in data:
            pos = str(data.get("dustbin_position")).strip().lower()
            if pos in ("center_left", "bucket"):
                DUSTBIN_ZONE_ENABLED = True
                DUSTBIN_ZONE = [140, 160, 390, 470]
            elif pos == "bottom_right":
                DUSTBIN_ZONE_ENABLED = True
                DUSTBIN_ZONE = [400, 250, 630, 470]
            elif pos == "bottom_left":
                DUSTBIN_ZONE_ENABLED = True
                DUSTBIN_ZONE = [10, 250, 240, 470]
            elif pos == "bottom_center":
                DUSTBIN_ZONE_ENABLED = True
                DUSTBIN_ZONE = [205, 250, 435, 470]
            elif pos in ("disabled", "none"):
                DUSTBIN_ZONE_ENABLED = False
                DUSTBIN_ZONE = None
        elif "dustbin_coords" in data and isinstance(data.get("dustbin_coords"), list) and len(data.get("dustbin_coords")) == 4:
            DUSTBIN_ZONE_ENABLED = True
            DUSTBIN_ZONE = [int(x) for x in data.get("dustbin_coords")]

        if camera_reconnect_needed:
            # Signal stream to reconnect with new config
            stream_stop_event.set()
            with camera_lock:
                if camera is not None:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    camera = None

        return jsonify({
            "success": True,
            "message": "Camera and Dustbin settings updated.",
            "camera_mode": CAMERA_MODE,
            "phone_url": PHONE_CAMERA_URL,
            "dustbin_zone_enabled": DUSTBIN_ZONE_ENABLED,
            "dustbin_zone": get_effective_dustbin_zone()
        })

    return jsonify({
        "success": True,
        "camera_mode": CAMERA_MODE,
        "phone_url": PHONE_CAMERA_URL,
        "camera_id": CAMERA_ID,
        "location": CAMERA_LOCATION,
        "dustbin_zone_enabled": DUSTBIN_ZONE_ENABLED,
        "dustbin_zone": get_effective_dustbin_zone()
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI WASTE DETECTION SYSTEM")
    print("=" * 60)

    print(
        "Camera ID:",
        CAMERA_ID
    )

    print(
        "Location:",
        CAMERA_LOCATION
    )

    print()

    print(
        "PERSON ONLY       -> SAVE PERSON DETECTION"
    )

    print(
        "PERSON ONLY       -> NO DUMPING"
    )

    print(
        "PERSON + FLOOR WASTE -> WASTE DETECTED / NO DUMPING"
    )

    print(
        "BOTTLE            -> COCO BOTTLE DETECTION"
    )

    print(
        "HAND + WASTE      -> HOLDING (only when waste is near wrist)"
    )

    print(
        "WASTE RELEASE     -> RELEASING"
    )

    print(
        "RELEASE CONFIRMED -> ILLEGAL DUMPING"
    )

    print(
        "OUT-OF-FRAME RELEASE -> ILLEGAL DUMPING AFTER CONFIRMATION"
    )
    print(
        "NO PERSON + NEW/MOVING WASTE -> ILLEGAL DUMPING"
    )
    print("UNKNOWN OBJECT MOTION FALLBACK: ENABLED (person-present and personless)")
    print("MOTION FALLBACK FILTERS: minimum blob 24x20 px, stronger multi-frame trajectory, 30s event cooldown")
    print(
        "PERSONLESS MODE requires", PERSONLESS_REQUIRED_EMPTY_FRAMES, "empty-person inference frames"
    )
    print(
        "Dustbin zone:",
        DUSTBIN_ZONE
    )

    print("WASTE MODEL CLASSES:")
    for class_id, class_name in get_custom_model_classes().items():
        print(f"  {class_id}: {class_name}")

    print("PERSON-INDEPENDENT DUMPING: ENABLED")
    if DUSTBIN_ZONE is None:
        print("Dumping region: FULL CAMERA FRAME")
    else:
        print("Dumping region:", DUSTBIN_ZONE)

    print()

    print(
        "Camera mode:", CAMERA_MODE
    )

    if CAMERA_MODE.upper() == "PHONE":
        print("Phone stream:", PHONE_CAMERA_URL)

    print()
    print(
        "Performance: inference every",
        INFERENCE_EVERY_N_FRAMES,
        "frames | imgsz",
        INFERENCE_IMAGE_SIZE,
        "| JPEG",
        JPEG_QUALITY
    )

    print()
    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5001"
    )

    print(
        "Evidence URL: http://127.0.0.1:5001/evidence/<filename>"
    )

    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True,
        use_reloader=False
    )
