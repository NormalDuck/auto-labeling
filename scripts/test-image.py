import cv2
from ultralytics.models.yolo import YOLO

# 1. Load your custom trained model
# Replace this path with the actual path to your weights file
model = YOLO("best.pt")

# 2. Run inference on a new image
# You can use a path to a single image, a folder, or even a URL
source_img = "validate.jpg"
results = model(source_img, conf=0.25)  # conf=0.25 ignores low-confidence guesses

# 3. Process the results
for r in results:
    # This will save the image with boxes drawn on it to 'runs/detect/predict/'
    r.save(filename="result_prediction.jpg")

    # Or, if you want to see the coordinates in your console:
    for box in r.boxes:
        # Get coordinates in [x1, y1, x2, y2] format
        coords = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        print(f"Detected Class {class_id} with {conf:.2f} confidence at {coords}")

print("Prediction saved as result_prediction.jpg")
