import os
import shutil

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

# --- 1. SETUP ---
INPUT_DIR = "pictures"  # Your source folder
DATASET_ROOT = "dataset"
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Replace with your actual key

# Ensure standard YOLO structure exists
os.makedirs(f"{DATASET_ROOT}/train/images", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/train/labels", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/val/images", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/val/labels", exist_ok=True)

client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)

# --- 2. ITERATE THROUGH DIRECTORY ---
# Get list of all image files in the directory
image_extensions = (".png", ".jpg", ".jpeg")
files_to_process = [
    f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)
]

print(f"Found {len(files_to_process)} images in '{INPUT_DIR}'. Starting annotation...")

for filename in files_to_process:
    image_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    # --- 3. RUN INFERENCE ---
    try:
        result = client.run_workflow(
            workspace_name="matrix-autolabeler",
            workflow_id="find-yellow-solid-balls",
            images={"image": image_path},
        )

        # --- 4. PROCESS & SAVE ---
        # Each 'output' in result corresponds to one image sent
        for output in result:
            predictions_dict = output.get("predictions", {})
            image_info = predictions_dict.get("image", {})
            img_w, img_h = image_info.get("width"), image_info.get("height")
            boxes_list = predictions_dict.get("predictions", [])

            yolo_lines = []
            for box in boxes_list:
                # Normalize coordinates for YOLO format
                x_norm = box.get("x") / img_w
                y_norm = box.get("y") / img_h
                w_norm = box.get("width") / img_w
                h_norm = box.get("height") / img_h

                yolo_lines.append(
                    f"{box.get('class_id')} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            # Move image to training folder
            shutil.copy(image_path, f"{DATASET_ROOT}/train/images/{filename}")

            # Save matching .txt label file
            with open(f"{DATASET_ROOT}/train/labels/{base_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"Successfully annotated: {filename}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# --- 5. CREATE data.yaml ---
yaml_content = f"""
path: {os.path.abspath(DATASET_ROOT)}
train: train/images
val: train/images

names:
  0: fuel
"""

with open(f"{DATASET_ROOT}/data.yaml", "w") as f:
    f.write(yaml_content)

print(f"\nProcessing complete. Dataset ready at: {DATASET_ROOT}")
