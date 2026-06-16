from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # lightweight

# Train
model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640
)