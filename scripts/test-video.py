# 1. Load your model
from ultralytics.models import YOLO

model = YOLO("best.pt")

# 2. Run inference on a video
# Ultralytics will automatically detect it's a video and process frame-by-frame
results = model.predict(source="nah.mp4", save=True, conf=0.25)

# The annotated video will be saved in 'runs/detect/predict/'
