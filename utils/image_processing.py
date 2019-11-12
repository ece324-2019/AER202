import os
from PIL import Image

FULL_IMAGE_DIR = '../datasets/full_raw/'
FULL_IMAGE_SAVE_DIR = '../datasets/full/'
TEST_IMAGE_DIR = '../datasets/test/'
TEST_IMAGE_SAVE_DIR = '../datasets/test_cropped/'
TARGET_ASPECT_RATIO = 1.75  # width / height
RESIZED_IMAGE_WIDTH = 100


def open_files(is_small_dataset):
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


def crop_and_rescale_images(image_array, is_small_dataset):
    """Crop and rescale images to ensure uniformity before using them in training"""
    images_save_dir = TEST_IMAGE_SAVE_DIR if is_small_dataset else FULL_IMAGE_SAVE_DIR

    for i in range(len(image_array)):
        image, plane_type_dir = image_array[i]
        width, height = image.size
        # ratio = w / h
        aspect_ratio = width / height
        # Image needs to be cropped horizontally to match target
        if aspect_ratio > TARGET_ASPECT_RATIO:
            # Calculate target width
            target_width = TARGET_ASPECT_RATIO * height
            width_difference = width - target_width
            # Crop equally on left and right
            left = width_difference / 2
            top = 0
            right = width - (width_difference / 2)
            bottom = height
            resized_image_height = int(
                RESIZED_IMAGE_WIDTH / TARGET_ASPECT_RATIO)
            processed_image = image.resize(
                (RESIZED_IMAGE_WIDTH, resized_image_height), box=(left, top, right, bottom))
        # Otherwise, image needs to be cropped vertically to match target
        else:
            # Calculate target height
            target_height = width / TARGET_ASPECT_RATIO
            height_difference = height - target_height
            # Crop equally on top and bottom
            left = 0
            top = height_difference / 2
            right = width
            bottom = height - (height_difference / 2)
            resized_image_height = int(
                RESIZED_IMAGE_WIDTH / TARGET_ASPECT_RATIO)
            processed_image = image.resize(
                (RESIZED_IMAGE_WIDTH, resized_image_height), box=(left, top, right, bottom))

        filename = str(i) + '.jpg'
        save_dir = os.path.join(images_save_dir, plane_type_dir, filename)
        # Create directory for new type of plane if it does not exist
        plane_type_path = os.path.join(images_save_dir, plane_type_dir)
        if not os.path.exists(plane_type_path):
            os.makedirs(plane_type_path)
        processed_image.save(save_dir)


def process_images(is_small_dataset):
    """Process images from datasets directory"""
    images = open_files(is_small_dataset)
    crop_and_rescale_images(images, is_small_dataset)
    close_files(images)
