import os
from PIL import Image

TEST_IMAGE_DIR = '../datasets/test/'
TEST_SAVE_IMAGE_DIR = '../datasets/test_cropped/'
TARGET_ASPECT_RATIO = 1.75  # width / height
RESIZED_IMAGE_WIDTH = 100


def open_files():
    # Array to store opened images
    image_array = []

    for root, dirs, files in os.walk(TEST_IMAGE_DIR, topdown=False):
        for name in files:
            image_array.append(Image.open(root + name))

    return image_array


def close_files(image_array):
    for image in image_array:
        image.close()


def crop_and_rescale_images(image_array):
    for i in range(len(image_array)):
        image = image_array[i]
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
            resized_image_height = int(RESIZED_IMAGE_WIDTH / TARGET_ASPECT_RATIO)
            processed_image = image.resize((RESIZED_IMAGE_WIDTH, resized_image_height), box=(left, top, right, bottom))
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
            resized_image_height = int(RESIZED_IMAGE_WIDTH / TARGET_ASPECT_RATIO)
            processed_image = image.resize((RESIZED_IMAGE_WIDTH, resized_image_height), box=(left, top, right, bottom))

        processed_image.save(TEST_SAVE_IMAGE_DIR + str(i) + '.jpg')


def load_images():
    images = open_files()
    crop_and_rescale_images(images)
    close_files(images)
