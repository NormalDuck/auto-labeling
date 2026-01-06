import locale
import os

from IPython.display import Image
from ultralytics import YOLO

HOME = os.getcwd()
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"
target_model = YOLO("yolo11n.pt")
target_model.train(DATA_YAML_PATH, epochs=50)


locale.getpreferredencoding = lambda: "UTF-8"


Image(filename=f"{HOME}/runs/detect/train/confusion_matrix.png", width=600)
