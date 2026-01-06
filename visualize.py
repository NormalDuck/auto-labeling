import os

import cv2
import matplotlib.pyplot as plt


def visualize_yolo(image_path, label_path, class_names):
    # 1. Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not decode image at {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # 2. Check if label exists
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found at {label_path}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # Extract first 5 elements, ignore any extra (like confidence)
        class_id = int(parts[0])
        x_c, y_c, bbox_w, bbox_h = map(float, parts[1:5])

        # Convert normalized to pixel coordinates
        xmin = int((x_c - bbox_w / 2) * w)
        ymin = int((y_c - bbox_h / 2) * h)
        xmax = int((x_c + bbox_w / 2) * w)
        ymax = int((y_c + bbox_h / 2) * h)

        # Draw
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = (
            class_names[class_id] if class_id < len(class_names) else f"ID: {class_id}"
        )
        cv2.putText(
            image,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    # plt.show()
    plt.savefig("output_test.jpg")


img = "week1-clips-00257"

# --- UPDATE THESE PATHS ---
# Example: Use absolute paths or relative paths from your current terminal directory
classes = ["coral", "algae"]
img_file = "dataset/train/images/" + img + ".jpg"  # Replace with a real filename
lbl_file = "dataset/train/labels/" + img + ".txt"  # Replace with a real filename

visualize_yolo(img_file, lbl_file, classes)

print("Visualization saved to output_test.jpg")
