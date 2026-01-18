# 1. Load a pretrained model (yolov8n or yolov11n are current standards)
# YOLOv12 would be loaded similarly once officially released in the Ultralytics stable branch
from ultralytics.models import YOLO

model = YOLO("yolo11n.pt")

# 2. Train the model
results = model.train(
    data="dataset/data.yaml",  # Path to your yaml file
    epochs=100,  # How many times to see the data
    # imgsz=640,  # Image size (standard for YOLO)
    batch=16,  # Adjust based on your GPU memory
)
