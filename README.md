# ♻️ AI Waste Detection & Illegal Dumping Monitoring System

Smart AI-powered computer vision system designed to detect waste and identify illegal garbage dumping in real time using YOLOv8/YOLO11, ByteTrack pose tracking, and Flask.

---

## 🌟 Key Features

1. **Real-time Waste Detection:**
   - Detects waste items using a custom fine-tuned YOLO model (`runs/detect/waste_detection_fixed/weights/best.pt`).
   - COCO fallback detector for plastic bottle detection (`yolo11n.pt`).
   - Foreground motion & trajectory tracking fallback (MOG2 background subtraction) for unknown thrown objects.

2. **Person & Pose Tracking:**
   - Detects human keypoints (wrists, elbows) using `yolo11n-pose.pt`.
   - Multi-person tracking with ByteTrack to track identity across video frames.

3. **Illegal Dumping Verification Engine:**
   - **Hand-to-Waste Holding Confirmation:** Tracks when a person is holding waste near their wrist for consecutive frames.
   - **Release / Drop Detection:** Detects when the held waste is separated from the wrist and moves away or is thrown out of frame.
   - **Person-Independent Dumping:** Detects thrown or newly appearing waste even when no person is visible in frame.

4. **Web Dashboard & User Interface:**
   - **Live Monitoring Dashboard (`/`):** Real-time camera feed, live alert cards, system statistics, and recent activity log.
   - **Dedicated Camera View (`/camera`):** Streamlined full-screen camera view with real-time stats.
   - **Detection History (`/history`):** Complete database log with search filter, type filter, evidence viewer, and **CSV Export** button.
   - **Static Image Upload & Test (`/detect`):** Upload any test picture directly from the dashboard to get instant YOLO bounding boxes and analysis.
   - **Dynamic Camera Source Switcher (`/api/camera_settings`):** Easily switch between PC webcam (0) and Phone IP Webcam (`http://...:8080/video`) directly from the UI.
   - **Real-Time Sound Alarm:** Plays an audio alert tone in the browser when an illegal dumping incident is detected.
   - **Automated Evidence Capture:** Annotated snapshots (`static/results/illegal_dumping_*.jpg`) saved with timestamps and bounding boxes.

---

## 📁 Project Structure

```
waste detection system/
├── app.py                      # Main Flask application and AI pipeline
├── live_camera.py              # Standalone OpenCV script for quick camera testing
├── camera_test.py              # Simple webcam test script
├── yolo_train.py               # Model training script for YOLO
├── yolo_test.py                # Model test script
├── convert_labels.py           # Dataset polygon to YOLO bbox converter
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── database/
│   └── waste_detection.db      # SQLite database storing detection history
├── static/
│   ├── results/                # Saved evidence screenshots and test results
│   └── uploads/                # Uploaded test images
├── templates/
│   ├── index.html              # Main dashboard UI
│   ├── camera.html             # Dedicated live camera page
│   └── history.html            # Detection history and audit table
├── runs/detect/
│   └── waste_detection_fixed/  # Custom trained YOLO waste detection weights
├── dataset/                    # Training dataset and data.yaml
├── yolo11n-pose.pt             # Ultralytics pose estimation model
└── yolo11n.pt                  # Ultralytics COCO model (bottle fallback)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ installed.
- (Optional) NVIDIA GPU with CUDA for faster inference.

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 📹 Camera Configuration

You can use either your **PC Webcam** or your **Smartphone Camera**.

### Option A: Using Phone Camera (Recommended)
1. Install **IP Webcam** (Android) or **EpocCam / DroidCam** from Play Store / App Store.
2. Open IP Webcam on your phone, scroll to the bottom and click **Start Server**.
3. Note the IP address shown on screen (e.g., `http://192.168.1.15:8080`).
4. Enter `http://<YOUR_PHONE_IP>:8080/video` into the dashboard's **⚙ Camera Source** modal or in `app.py` under `PHONE_CAMERA_URL`.

### Option B: Using PC Webcam
1. Open the dashboard and click **⚙ Camera Source**.
2. Select **PC / USB Webcam** and save.

---

## 🏃 Running the Application

Start the Flask server:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard page |
| `/camera` | GET | Dedicated camera view page |
| `/video_feed` | GET | Multipart MJPEG live video stream |
| `/status` | GET | Current live AI tracking and detection status (JSON) |
| `/api/latest-alert` | GET | Most recent illegal dumping incident alert (JSON) |
| `/api/alerts` | GET | Recent 50 illegal dumping records (JSON) |
| `/detect` | POST | Upload an image file for immediate YOLO analysis |
| `/history` | GET | Detection history and evidence viewer page |
| `/api/history` | GET | All recorded detection logs (JSON) |
| `/export_history_csv`| GET | Download entire detection database as a `.csv` file |
| `/api/camera_settings`| GET / POST | View or update camera mode (`PC` vs `PHONE`) and URL |
| `/stop_camera` | POST | Safely stop the camera feed and reset state |
| `/clear_history` | POST | Clear all detections from SQLite database |
| `/health` | GET | System health check (models, camera, database status) |

---

## 📄 License
Educational and environmental monitoring project.

