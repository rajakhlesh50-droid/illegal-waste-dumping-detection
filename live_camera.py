import os
import cv2
from ultralytics import YOLO

# Trained YOLO model
MODEL_PATH = os.path.join("runs", "detect", "waste_detection_fixed", "weights", "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("runs", "detect", "runs", "waste_detection", "weights", "best.pt")

model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha!")
    exit()

print("✅ Live Waste Detection Started")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera frame nahi mila")
        break

    # YOLO detection
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show camera
    cv2.imshow(
        "Live Waste Detection - YOLO",
        annotated_frame
    )

    # Q = Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")