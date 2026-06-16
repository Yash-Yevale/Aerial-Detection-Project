from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Test on image
results = model("yolo_dataset/test/images", show=True)