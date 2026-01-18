import os

import cv2

# --- CONFIGURATION ---
DATASET_ROOT = "dataset"
IMAGES_DIR = os.path.join(DATASET_ROOT, "train/images")
LABELS_DIR = os.path.join(DATASET_ROOT, "train/labels")
OUTPUT_DIR = "visualized_annotations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a color for our class (Green)
COLOR = (0, 255, 0)

# --- PROCESSING ---
# Get all image files
image_files = [
    f for f in os.listdir(IMAGES_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
]

for img_file in image_files:
    # 1. Load the image
    img_path = os.path.join(IMAGES_DIR, img_file)
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    # 2. Find the corresponding label file
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_file)

    if not os.path.exists(label_path):
        print(f"Warning: No label found for {img_file}")
        continue

    # 3. Read YOLO annotations
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id, x_norm, y_norm, w_norm, h_norm = map(float, parts)

        # 4. De-normalize (Convert 0.0-1.0 back to Pixels)
        # YOLO format is center-based
        x_center = x_norm * width
        y_center = y_norm * height
        w = w_norm * width
        h = h_norm * height

        # Convert Center to Top-Left and Bottom-Right for OpenCV
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # 5. Draw the box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), COLOR, 2)
        cv2.putText(
            image,
            f"ID:{int(class_id)}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR,
            2,
        )

    # 6. Save the visualized image
    output_path = os.path.join(OUTPUT_DIR, f"vis_{img_file}")
    cv2.imwrite(output_path, image)
    print(f"Saved visualization: {output_path}")

print(f"\nAll visualizations are in: {OUTPUT_DIR}")
