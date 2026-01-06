import os
from pathlib import Path

import cv2
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from tqdm.notebook import tqdm

HOME = os.getcwd()
print(HOME)

os.makedirs("images", exist_ok=True)

VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10


video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH, extensions=["mov", "mp4"]
)

TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[:2], video_paths[2:]

for video_path in tqdm(TRAIN_VIDEO_PATHS):
    video_name = video_path.stem
    image_name_pattern = video_name + "-{:05d}.png"
    with sv.ImageSink(
        target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern
    ) as sink:
        for image in sv.get_video_frames_generator(
            source_path=str(video_path), stride=FRAME_STRIDE
        ):
            sink.save_image(image=image)


image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH, extensions=["png", "jpg", "jpg"]
)

print("image count:", len(image_paths))

IMAGE_DIR_PATH = f"{HOME}/images"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)


titles = [image_path.stem for image_path in image_paths[:SAMPLE_SIZE]]
images = [cv2.imread(str(image_path)) for image_path in image_paths[:SAMPLE_SIZE]]

sv.plot_images_grid(
    images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE
)


ontology = CaptionOntology(
    {
        # Algae Prompts
        "teal dimpled rubber ball": "algae",
        "cyan textured sphere": "algae",
        # Coral Prompts
        "white hollow pvc pipe": "coral",
        "short white plastic tube": "coral",
        "white cylinder with red lettering": "coral",
    }
)

DATASET_DIR_PATH = f"{HOME}/dataset"


base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH, extension=".png", output_folder=DATASET_DIR_PATH
)

ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH,
)

len(dataset)


mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

images = []
image_names = []
for i, (image_path, image, annotation) in enumerate(dataset):
    if i == SAMPLE_SIZE:
        break
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image, detections=annotation
    )
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=annotation
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=annotation
    )

    image_names.append(Path(image_path).name)
    images.append(annotated_image)

sv.plot_images_grid(
    images=images, titles=image_names, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE
)
