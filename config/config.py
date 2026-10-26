import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# CAMERA
# ============================================================

CAMERA_MODE = "PHONE"

PHONE_CAMERA_URL = "http://10.124.14.213:8080/video"

CAMERA_ID = "CAM-01"
CAMERA_LOCATION = "Paralakhemundi, Odisha"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

CAMERA_BUFFER_SIZE = 1
CAMERA_READ_FAILURE_LIMIT = 30
CAMERA_RECONNECT_DELAY = 0.5

# ============================================================
# AI MODELS
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
# DETECTION
# ============================================================

WASTE_CONFIDENCE = 0.32
PERSON_CONFIDENCE = 0.50
BOTTLE_CONFIDENCE = 0.35

INFERENCE_EVERY_N_FRAMES = 2
INFERENCE_IMAGE_SIZE = 512
JPEG_QUALITY = 65

# ============================================================
# HOLDING / RELEASE
# ============================================================

REQUIRED_HOLDING_FRAMES = 2
REQUIRED_RELEASE_FRAMES = 2

HAND_WASTE_DISTANCE = 90
RELEASE_DISTANCE = 85

MIN_RELEASE_MOTION = 12
FAST_RELEASE_MOTION = 25
MIN_DOWNWARD_DROP_MOTION = 10

NO_PERSON_RELEASE_GRACE_FRAMES = 6

# ============================================================
# ILLEGAL DUMPING
# ============================================================

DUMPING_EVENT_COOLDOWN = 10
ALERT_VISIBLE_SECONDS = 8
LEGAL_ALERT_VISIBLE_SECONDS = 8

# Person-independent dumping
PERSON_INDEPENDENT_DUMPING = False

# ============================================================
# DATABASE
# ============================================================

DATABASE_FOLDER = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE = os.path.join(
    DATABASE_FOLDER,
    "waste_detection.db"
)

# ============================================================
# STORAGE
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

os.makedirs(
    DATABASE_FOLDER,
    exist_ok=True
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)