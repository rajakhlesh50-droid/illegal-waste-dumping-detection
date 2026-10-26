import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2
from flask import (
    Flask,
    render_template,
    request,
    Response,
    redirect,
    url_for,
    jsonify
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename


# ============================================================
# BASE CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR, "static", "results"
)

DATABASE_FOLDER = os.path.join(
    BASE_DIR, "database"
)

DATABASE = os.path.join(
    DATABASE_FOLDER,
    "waste_detection.db"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)


# ============================================================
# MODEL PATHS
# ============================================================

WASTE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "waste_detection_fixed",
    "weights",
    "best.pt"
)

PERSON_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE = 0.50
PERSON_CONFIDENCE = 0.45

CAMERA_ID = "CAM-01"
CAMERA_LOCATION = "Paralakhemundi, Odisha"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_SAVE_INTERVAL = 5

REQUIRED_INTERACTION_FRAMES = 10
INTERACTION_END_GRACE_FRAMES = 10

INTERACTION_DISTANCE_RATIO = 0.35

DUMPING_EVENT_COOLDOWN = 30


# ============================================================
# GLOBAL CAMERA
# ============================================================

camera = None
camera_lock = threading.Lock()


# ============================================================
# TEMPORAL STATE
# ============================================================

interaction_streak = 0
interaction_end_streak = 0

interaction_confirmed = False
dumping_event_active = False

last_dumping_event_time = 0

last_waste_boxes = []
last_person_boxes = []

last_interaction_frame = None


# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
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
            evidence_image TEXT
        )
    """)

    columns = [
        row["name"]
        for row in cursor.execute(
            "PRAGMA table_info(detections)"
        ).fetchall()
    ]

    if "person_detected" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN person_detected INTEGER DEFAULT 0
        """)

    if "illegal_dumping" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN illegal_dumping INTEGER DEFAULT 0
        """)

    if "interaction_detected" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN interaction_detected INTEGER DEFAULT 0
        """)

    if "camera_id" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN camera_id TEXT DEFAULT 'CAM-01'
        """)

    if "location" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN location TEXT DEFAULT 'Unknown'
        """)

    if "evidence_image" not in columns:

        cursor.execute("""
            ALTER TABLE detections
            ADD COLUMN evidence_image TEXT
        """)

    conn.commit()

    conn.close()


init_database()


# ============================================================
# LOAD WASTE MODEL
# ============================================================

print()
print("==========================================")
print("Loading Waste Detection Model...")
print("==========================================")

if not os.path.exists(WASTE_MODEL_PATH):

    raise FileNotFoundError(
        f"Waste model not found:\n{WASTE_MODEL_PATH}"
    )

waste_model = YOLO(WASTE_MODEL_PATH)

print("Waste Model Loaded Successfully")
print("Waste Classes:", waste_model.names)


# ============================================================
# LOAD PERSON MODEL
# ============================================================

print()
print("==========================================")
print("Loading Person Detection Model...")
print("==========================================")

if not os.path.exists(PERSON_MODEL_PATH):

    raise FileNotFoundError(
        f"Person model not found:\n{PERSON_MODEL_PATH}"
    )

person_model = YOLO(PERSON_MODEL_PATH)

print("Person Model Loaded Successfully")
print("Person Class:", person_model.names[0])


# ============================================================
# SAVE DETECTION
# ============================================================

def save_detection(
    waste_type,
    confidence,
    source,
    person_detected=False,
    illegal_dumping=False,
    interaction_detected=False,
    camera_id=CAMERA_ID,
    location=CAMERA_LOCATION,
    evidence_image=None
):

    conn = get_db()

    cursor = conn.cursor()

    detected_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO detections
        (
            waste_type,
            confidence,
            source,
            detected_at,
            person_detected,
            illegal_dumping,
            interaction_detected,
            camera_id,
            location,
            evidence_image
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        waste_type,
        float(confidence),
        source,
        detected_at,
        int(person_detected),
        int(illegal_dumping),
        int(interaction_detected),
        camera_id,
        location,
        evidence_image
    ))

    conn.commit()

    conn.close()


# ============================================================
# PERSON DETECTION
# ============================================================

def detect_persons(frame):

    results = person_model.predict(
        source=frame,
        conf=PERSON_CONFIDENCE,
        classes=[0],
        verbose=False
    )

    persons = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = xyxy

            persons.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })

    return persons


# ============================================================
# WASTE DETECTION
# ============================================================

def detect_waste(frame):

    results = waste_model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    wastes = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )

            class_name = waste_model.names.get(
                class_id,
                "waste"
            )

            x1, y1, x2, y2 = xyxy

            wastes.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            })

    return wastes


# ============================================================
# BOX CENTER
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return (
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    ) ** 0.5


# ============================================================
# PERSON + WASTE INTERACTION
# ============================================================

def check_interaction(
    persons,
    wastes,
    frame_width,
    frame_height
):

    if not persons or not wastes:

        return False

    max_dimension = max(
        frame_width,
        frame_height
    )

    distance_threshold = (
        max_dimension *
        INTERACTION_DISTANCE_RATIO
    )

    for person in persons:

        person_center = box_center(
            person["box"]
        )

        for waste in wastes:

            waste_center = box_center(
                waste["box"]
            )

            distance = calculate_distance(
                person_center,
                waste_center
            )

            if distance <= distance_threshold:

                return True

    return False


# ============================================================
# DRAW PERSON
# ============================================================

def draw_person(frame, person):

    x1, y1, x2, y2 = person["box"]

    confidence = person["confidence"]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    label = (
        f"PERSON {confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


# ============================================================
# DRAW WASTE
# ============================================================

def draw_waste(frame, waste):

    x1, y1, x2, y2 = waste["box"]

    confidence = waste["confidence"]

    class_name = waste["class_name"]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"{class_name.upper()} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# ============================================================
# RESET STATE
# ============================================================

def reset_event_state():

    global interaction_streak
    global interaction_end_streak
    global interaction_confirmed
    global dumping_event_active
    global last_waste_boxes
    global last_person_boxes
    global last_interaction_frame

    interaction_streak = 0
    interaction_end_streak = 0

    interaction_confirmed = False
    dumping_event_active = False

    last_waste_boxes = []
    last_person_boxes = []

    last_interaction_frame = None


# ============================================================
# CREATE EVIDENCE IMAGE
# ============================================================

def create_evidence_image(
    frame,
    persons,
    wastes
):

    if frame is None:

        return None

    evidence_frame = frame.copy()

    for person in persons:

        draw_person(
            evidence_frame,
            person
        )

    for waste in wastes:

        draw_waste(
            evidence_frame,
            waste
        )

    cv2.putText(
        evidence_frame,
        "SUSPECTED ILLEGAL DUMPING",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        evidence_frame,
        f"Camera: {CAMERA_ID}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        evidence_frame,
        f"Location: {CAMERA_LOCATION}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        evidence_frame,
        datetime.now().strftime(
            "Time: %Y-%m-%d %H:%M:%S"
        ),
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    evidence_name = (
        "illegal_dumping_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".jpg"
    )

    evidence_path = os.path.join(
        RESULT_FOLDER,
        evidence_name
    )

    success = cv2.imwrite(
        evidence_path,
        evidence_frame
    )

    if not success:

        print(
            "ERROR: Could not save evidence image."
        )

        return None

    print(
        f"Evidence saved: {evidence_path}"
    )

    return evidence_name


# ============================================================
# ILLEGAL DUMPING PROCESSOR
# ============================================================

def process_illegal_dumping(
    interaction,
    persons,
    wastes,
    frame
):

    global interaction_streak
    global interaction_end_streak
    global interaction_confirmed
    global dumping_event_active
    global last_dumping_event_time
    global last_waste_boxes
    global last_person_boxes
    global last_interaction_frame

    current_time = time.time()

    # --------------------------------------------------------
    # INTERACTION PRESENT
    # --------------------------------------------------------

    if interaction:

        interaction_end_streak = 0

        interaction_streak += 1

        last_waste_boxes = [
            waste["box"]
            for waste in wastes
        ]

        last_person_boxes = [
            person["box"]
            for person in persons
        ]

        last_interaction_frame = frame.copy()

        if (
            interaction_streak >=
            REQUIRED_INTERACTION_FRAMES
        ):

            if not interaction_confirmed:

                interaction_confirmed = True

                print()
                print("================================")
                print("PERSON + WASTE INTERACTION")
                print(
                    f"Camera: {CAMERA_ID}"
                )
                print(
                    f"Location: {CAMERA_LOCATION}"
                )
                print(
                    f"Persons: {len(persons)}"
                )
                print(
                    f"Waste: {len(wastes)}"
                )
                print(
                    "Interaction Confirmed"
                )
                print(
                    "Time:",
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
                print("================================")

    # --------------------------------------------------------
    # INTERACTION ENDED
    # --------------------------------------------------------

    else:

        if interaction_confirmed:

            interaction_end_streak += 1

            if (
                interaction_end_streak >=
                INTERACTION_END_GRACE_FRAMES
            ):

                if (
                    current_time -
                    last_dumping_event_time
                    >=
                    DUMPING_EVENT_COOLDOWN
                ):

                    if not dumping_event_active:

                        dumping_event_active = True

                        last_dumping_event_time = (
                            current_time
                        )

                        detected_at = (
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        )

                        print()
                        print()
                        print(
                            "########################################"
                        )
                        print(
                            "🚨 SUSPECTED ILLEGAL DUMPING DETECTED"
                        )
                        print(
                            "########################################"
                        )
                        print(
                            f"Camera ID : {CAMERA_ID}"
                        )
                        print(
                            f"Location  : {CAMERA_LOCATION}"
                        )
                        print(
                            f"Time      : {detected_at}"
                        )
                        print(
                            f"Persons   : "
                            f"{len(last_person_boxes)}"
                        )
                        print(
                            f"Waste     : "
                            f"{len(last_waste_boxes)}"
                        )
                        print(
                            "Status    : "
                            "SUSPECTED ILLEGAL DUMPING"
                        )
                        print(
                            "########################################"
                        )
                        print()

                        # ------------------------------------
                        # EVIDENCE FRAME
                        # ------------------------------------

                        evidence_frame = (
                            last_interaction_frame
                            if last_interaction_frame is not None
                            else frame
                        )

                        evidence_persons = []

                        evidence_wastes = []

                        if last_person_boxes:

                            for person_box in last_person_boxes:

                                evidence_persons.append({
                                    "box": person_box,
                                    "confidence": 0.0
                                })

                        if last_waste_boxes:

                            for waste_box in last_waste_boxes:

                                evidence_wastes.append({
                                    "box": waste_box,
                                    "confidence": 0.0,
                                    "class_name": "waste"
                                })

                        evidence_name = create_evidence_image(
                            evidence_frame,
                            evidence_persons,
                            evidence_wastes
                        )

                        # ------------------------------------
                        # SAVE DATABASE
                        # ------------------------------------

                        if wastes:

                            for waste in wastes:

                                save_detection(
                                    waste_type=waste[
                                        "class_name"
                                    ],
                                    confidence=waste[
                                        "confidence"
                                    ],
                                    source="camera",
                                    person_detected=True,
                                    illegal_dumping=True,
                                    interaction_detected=True,
                                    camera_id=CAMERA_ID,
                                    location=CAMERA_LOCATION,
                                    evidence_image=evidence_name
                                )

                        else:

                            save_detection(
                                waste_type="waste",
                                confidence=0.0,
                                source="camera",
                                person_detected=True,
                                illegal_dumping=True,
                                interaction_detected=True,
                                camera_id=CAMERA_ID,
                                location=CAMERA_LOCATION,
                                evidence_image=evidence_name
                            )

                interaction_streak = 0
                interaction_end_streak = 0

                interaction_confirmed = False
                dumping_event_active = False

                last_waste_boxes = []
                last_person_boxes = []

                last_interaction_frame = None

        else:

            interaction_streak = 0


# ============================================================
# CAMERA
# ============================================================

def get_camera():

    global camera

    with camera_lock:

        if (
            camera is None
            or not camera.isOpened()
        ):

            camera = cv2.VideoCapture(0)

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
                20
            )

        return camera


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        detection_count=0,
        detections=[],
        result_image=None,
        input_image=None,
        person_count=0,
        interaction_detected=False,
        illegal_dumping=False,
        camera_id=CAMERA_ID,
        location=CAMERA_LOCATION
    )


# ============================================================
# CAMERA PAGE
# ============================================================

@app.route("/camera")
def camera_page():

    return render_template(
        "camera.html",
        camera_id=CAMERA_ID,
        location=CAMERA_LOCATION
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:

        return redirect(
            url_for("index")
        )

    file = request.files["image"]

    if file.filename == "":

        return redirect(
            url_for("index")
        )

    filename = secure_filename(
        file.filename
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    input_filename = (
        f"input_{timestamp}_{filename}"
    )

    input_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename
    )

    file.save(input_path)

    frame = cv2.imread(input_path)

    if frame is None:

        return "Invalid image", 400

    wastes = detect_waste(frame)

    persons = detect_persons(frame)

    person_detected = (
        len(persons) > 0
    )

    interaction = check_interaction(
        persons,
        wastes,
        frame.shape[1],
        frame.shape[0]
    )

    for person in persons:

        draw_person(
            frame,
            person
        )

    for waste in wastes:

        draw_waste(
            frame,
            waste
        )

        save_detection(
            waste_type=waste["class_name"],
            confidence=waste["confidence"],
            source="image",
            person_detected=person_detected,
            illegal_dumping=False,
            interaction_detected=interaction,
            camera_id="IMAGE",
            location="Uploaded Image",
            evidence_image=None
        )

    if interaction:

        status_text = (
            "PERSON + WASTE INTERACTION"
        )

    elif person_detected:

        status_text = (
            "PERSON DETECTED"
        )

    elif wastes:

        status_text = (
            "WASTE DETECTED"
        )

    else:

        status_text = (
            "NO DETECTION"
        )

    cv2.putText(
        frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Persons: {len(persons)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Waste: {len(wastes)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    result_filename = (
        f"result_{timestamp}_{filename}"
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        frame
    )

    return render_template(
        "index.html",
        detection_count=len(wastes),
        detections=wastes,
        result_image=result_filename,
        input_image=input_filename,
        person_count=len(persons),
        interaction_detected=interaction,
        illegal_dumping=False,
        camera_id="IMAGE",
        location="Uploaded Image"
    )


# ============================================================
# LIVE VIDEO
# ============================================================

def generate_frames():

    global camera

    last_camera_save = 0

    cam = get_camera()

    if cam is None or not cam.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        return

    while True:

        success, frame = cam.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        persons = detect_persons(frame)

        wastes = detect_waste(frame)

        interaction = check_interaction(
            persons,
            wastes,
            frame.shape[1],
            frame.shape[0]
        )

        process_illegal_dumping(
            interaction,
            persons,
            wastes,
            frame
        )

        for person in persons:

            draw_person(
                frame,
                person
            )

        for waste in wastes:

            draw_waste(
                frame,
                waste
            )

        if dumping_event_active:

            status_text = (
                "SUSPECTED ILLEGAL DUMPING"
            )

        elif interaction_confirmed:

            status_text = (
                "INTERACTION CONFIRMED"
            )

        elif interaction:

            status_text = (
                "PERSON + WASTE NEARBY"
            )

        elif persons:

            status_text = (
                "PERSON DETECTED"
            )

        elif wastes:

            status_text = (
                "WASTE DETECTED"
            )

        else:

            status_text = (
                "MONITORING..."
            )

        cv2.putText(
            frame,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Persons: {len(persons)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Waste: {len(wastes)}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Camera: {CAMERA_ID}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Location: {CAMERA_LOCATION}",
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        current_time = time.time()

        if (
            wastes
            and current_time -
            last_camera_save
            >= CAMERA_SAVE_INTERVAL
        ):

            for waste in wastes:

                save_detection(
                    waste_type=waste["class_name"],
                    confidence=waste["confidence"],
                    source="camera",
                    person_detected=(
                        len(persons) > 0
                    ),
                    illegal_dumping=False,
                    interaction_detected=interaction,
                    camera_id=CAMERA_ID,
                    location=CAMERA_LOCATION,
                    evidence_image=None
                )

            last_camera_save = current_time

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:

            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )


# ============================================================
# STOP CAMERA
# ============================================================

@app.route("/stop_camera")
def stop_camera():

    global camera

    with camera_lock:

        if camera is not None:

            camera.release()

            camera = None

    reset_event_state()

    return redirect(
        url_for("index")
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
    """)

    detections = cursor.fetchall()

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
    """)

    total = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE source = 'camera'
    """)

    camera_count = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE source = 'image'
    """)

    image_count = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE interaction_detected = 1
    """)

    interaction_count = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE person_detected = 1
    """)

    person_count = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE illegal_dumping = 1
    """)

    illegal_dumping_count = (
        cursor.fetchone()["count"]
    )

    conn.close()

    return render_template(
        "history.html",
        detections=detections,
        total=total,
        camera_count=camera_count,
        image_count=image_count,
        interaction_count=interaction_count,
        person_count=person_count,
        illegal_dumping_count=illegal_dumping_count
    )


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history")
def api_history():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    data = [
        dict(row)
        for row in rows
    ]

    conn.close()

    return jsonify(data)


# ============================================================
# ALERT API
# ============================================================

@app.route("/api/alerts")
def api_alerts():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            waste_type,
            confidence,
            source,
            detected_at,
            person_detected,
            illegal_dumping,
            interaction_detected,
            camera_id,
            location,
            evidence_image
        FROM detections
        WHERE illegal_dumping = 1
        ORDER BY id DESC
        LIMIT 10
    """)

    rows = cursor.fetchall()

    alerts = [
        dict(row)
        for row in rows
    ]

    conn.close()

    return jsonify({
        "success": True,
        "count": len(alerts),
        "alerts": alerts
    })


# ============================================================
# LATEST ALERT
# ============================================================

@app.route("/api/latest-alert")
def latest_alert():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            waste_type,
            confidence,
            source,
            detected_at,
            person_detected,
            illegal_dumping,
            interaction_detected,
            camera_id,
            location,
            evidence_image
        FROM detections
        WHERE illegal_dumping = 1
        ORDER BY id DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    conn.close()

    if row is None:

        return jsonify({
            "success": True,
            "alert": None
        })

    return jsonify({
        "success": True,
        "alert": dict(row)
    })


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route("/clear_history")
def clear_history():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM detections"
    )

    conn.commit()

    conn.close()

    return redirect(
        url_for("history")
    )


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
    """)

    total = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE illegal_dumping = 1
    """)

    illegal_dumping_count = (
        cursor.fetchone()["count"]
    )

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE interaction_detected = 1
    """)

    interaction_count = (
        cursor.fetchone()["count"]
    )

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE person_detected = 1
    """)

    person_count = (
        cursor.fetchone()["count"]
    )

    conn.close()

    return jsonify({
        "system": "AI Waste Detection System",
        "status": "running",
        "camera_id": CAMERA_ID,
        "location": CAMERA_LOCATION,
        "total_detections": total,
        "person_detections": person_count,
        "interactions": interaction_count,
        "suspected_illegal_dumping": illegal_dumping_count
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    camera_status = False

    if camera is not None:

        camera_status = camera.isOpened()

    return jsonify({
        "status": "healthy",
        "waste_model": os.path.exists(
            WASTE_MODEL_PATH
        ),
        "person_model": os.path.exists(
            PERSON_MODEL_PATH
        ),
        "database": os.path.exists(
            DATABASE
        ),
        "camera_open": camera_status,
        "camera_id": CAMERA_ID,
        "location": CAMERA_LOCATION
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("     AI WASTE DETECTION SYSTEM")
    print("==========================================")
    print()

    print(
        f"Camera ID : {CAMERA_ID}"
    )

    print(
        f"Location  : {CAMERA_LOCATION}"
    )

    print()

    print(
        "Dashboard : http://127.0.0.1:5000"
    )

    print(
        "Camera    : http://127.0.0.1:5000/camera"
    )

    print(
        "History   : http://127.0.0.1:5000/history"
    )

    print(
        "Alerts    : http://127.0.0.1:5000/api/alerts"
    )

    print(
        "Health    : http://127.0.0.1:5000/health"
    )

    print(
        "Status    : http://127.0.0.1:5000/status"
    )

    print()
    print("==========================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )