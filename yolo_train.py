from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0,
    project="runs",
    name="waste_detection",
    exist_ok=True
)

print("Training Completed!")
print("Best model: runs/waste_detection/weights/best.pt")