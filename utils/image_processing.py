import os
from PIL import Image
import numpy as np
import cv2

# Paths are relative to main.py (i.e. where function is called)
FULL_IMAGE_DIR = 'datasets/full_raw/'
FULL_IMAGE_SAVE_DIR = 'datasets/full_yolo/'
TEST_IMAGE_DIR = 'datasets/test/'
TEST_IMAGE_SAVE_DIR = 'datasets/test_cropped/'
TARGET_ASPECT_RATIO = 1.75  # width / height
RESIZED_IMAGE_DIMENSION = 350
YOLO_BASE_DIR = 'models/yolo-coco/'


def open_files(is_small_dataset: bool = False):
    """Open all images to be processed and store them in an array"""

    # Array to store opened images
    image_array = []
    images_dir = TEST_IMAGE_DIR if is_small_dataset else FULL_IMAGE_DIR

    for root, dirs, files in os.walk(images_dir, topdown=True):
        # Ignore files that are not images
        files = [file for file in files if file.endswith(('.jpg', '.webp'))]
        for name in files:
            path = os.path.join(root, name)
            # Extract name of parent directory, which is the type of plane
            plane_type_dir = os.path.split(os.path.dirname(path))[1]
            # Create tuple with image and its parent directory name (i.e. type of plane)
            image_array.append((Image.open(path), plane_type_dir))

    return image_array


def close_files(image_array):
    """Close all images that were opened"""
    for image, _ in image_array:
        image.close()


def yolo_airplane_crop(image_array, padding: int = 20, confidence: float = 0.7, threshold: float = 0.3):
    """Use YOLO to identify the location of an aircraft in an image in order
    to crop out the background surrounding the aircraft"""

    # load COCO class labels YOLO model was trained on
    labels_dir = os.path.sep.join([YOLO_BASE_DIR, "coco.names"])
    labels = open(labels_dir).read().strip().split("\n")

    config_dir = os.path.sep.join([YOLO_BASE_DIR, "yolov3.cfg"])
    weights_dir = os.path.sep.join([YOLO_BASE_DIR, "yolov3.weights"])

    # Load YOLO object detector
    net = cv2.dnn.readNetFromDarknet(config_dir, weights_dir)

    image_crops = []

    for photo_data in image_array:
        image = photo_data[0]
        original_width, original_height = image.size
        width, height = image.size
        image = np.array(image)

        # Get output layers from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        boxes = []
        confidence_scores = []
        class_ids = []

        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence_score = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence_score > confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidence_scores.append(float(confidence_score))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidence_scores, confidence, threshold).flatten()

        if len(idxs) > 0:
            if labels[class_ids[0]] == 'aeroplane':
                (x, y) = (boxes[0][0], boxes[0][1])
                (w, h) = (boxes[0][2], boxes[0][3])
                # TODO: fix "box can't exceed original image size" error
                left = x - padding if x - padding > 0 else 0
                top = y - padding if y - padding > 0 else 0
                right = x + w + padding if x + w + padding < original_width else original_width
                bottom = y + h + padding if y + h + padding < original_height else original_height
                image_crops.append((left, top, right, bottom))
                print(left, top, right, bottom, bottom - top)
            else:
                print("ERROR: aeroplane not found")
                image_crops.append(None)

    return image_crops


def crop_and_rescale_images(image_array, image_crops, is_small_dataset: bool = False):
    """Crop and rescale images to ensure uniformity before using them in training"""

    images_save_dir = TEST_IMAGE_SAVE_DIR if is_small_dataset else FULL_IMAGE_SAVE_DIR

    for i in range(len(image_array)):
        image, plane_type_dir = image_array[i]
        width, height = image.size
        processed_image = image.crop(image_crops[i]).resize((RESIZED_IMAGE_DIMENSION, RESIZED_IMAGE_DIMENSION))

        filename = str(i) + '.jpg'
        save_dir = os.path.join(images_save_dir, plane_type_dir, filename)
        # Create directory for new type of plane if it does not exist
        plane_type_path = os.path.join(images_save_dir, plane_type_dir)
        if not os.path.exists(plane_type_path):
            os.makedirs(plane_type_path)
        processed_image.save(save_dir)


def process_images(is_small_dataset: bool = False):
    """Process images from datasets directory"""
    images = open_files(is_small_dataset)
    crops = yolo_airplane_crop(images, 20, 0.7, 0.3)
    crop_and_rescale_images(images, crops, is_small_dataset)
    close_files(images)
