import glob
import os

import cv2

# --- CONFIGURATION ---
# Path to the folder containing your videos
VIDEO_INPUT_DIR = "videos"
# Root folder for all extracted frames
FRAMES_OUTPUT_ROOT = "pictures"
# Save every Nth frame
FRAME_STRIDE = 10
# Supported video formats
EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

os.makedirs(FRAMES_OUTPUT_ROOT, exist_ok=True)

# Find all videos in the directory
video_files = []
for ext in EXTENSIONS:
    video_files.extend(glob.glob(os.path.join(VIDEO_INPUT_DIR, ext)))

if not video_files:
    print(f"No videos found in {VIDEO_INPUT_DIR}")
    exit()

print(f"Found {len(video_files)} videos. Starting extraction...")

for video_path in video_files:
    video_filename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_filename)[0]

    # Create a unique subfolder for each video to avoid overwriting frames
    video_output_dir = FRAMES_OUTPUT_ROOT

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    print(f"Processing: {video_filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_STRIDE == 0:
            # Name format: videoName_frameNumber.png
            img_name = f"{video_name_no_ext}_f{saved_count:05d}.png"
            save_path = os.path.join(video_output_dir, img_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to {video_output_dir}")

print("\nAll videos processed successfully.")
