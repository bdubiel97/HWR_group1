import os
from math import floor

import cv2 as cv
import numpy as np

WHITE = [255, 255, 255]

kernel = np.ones((3, 3), np.uint8)


def pad(image, output_size):
    size_x, size_y, _ = image.shape

    # Calculate center of gravity
    row_indices = [x for x in range(size_x) for y in range(size_y) if not any(image[x][y])]
    row_indices.sort()
    mean_x = row_indices[floor(len(row_indices) / 2)]
    col_indices = [y for x in range(size_x) for y in range(size_y) if not any(image[x][y])]
    col_indices.sort()
    mean_y = col_indices[floor(len(col_indices) / 2)]

    # Pad image to output_size
    pos_x = int(max(0, min(output_size[0] - size_x, output_size[0] / 2 - mean_x)))
    pos_y = int(max(0, min(output_size[1] - size_y, output_size[1] / 2 - mean_y)))

    image = cv.copyMakeBorder(image, pos_x, output_size[0] - size_x - pos_x, pos_y, output_size[1] - size_y - pos_y,
                              cv.BORDER_CONSTANT, value=WHITE)
    return image


def prepare_for_padding(image, output_size):
    size_x, size_y, _ = image.shape
    # Resize if necessary
    if size_x > output_size[0] or size_y > output_size[1]:
        image = cv.resize(image, (floor(size_y / 2), floor(size_x / 2)))

    return pad(image=image, output_size=output_size)


def pre_process(path, output_path, output_size=None):
    if not output_size:
        output_size = [128, 128]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Matrix 1 for shearing
    M1 = np.float32([[1, -0.2, 0], [0, 1, 0]])
    M1[0, 2] = -M1[0, 1] * output_size[0] / 2
    M1[1, 2] = -M1[1, 0] * output_size[1] / 2
    # Matrix 2 for shearing
    M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
    M2[0, 2] = -M2[0, 1] * output_size[0] / 2
    M2[1, 2] = -M2[1, 0] * output_size[1] / 2

    for style in os.listdir(path):
        output_folder = os.path.join(output_path, style)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for letter in os.listdir(os.path.join(path, style)):
            for image_name in os.listdir(os.path.join(path, style, letter)):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(path, style, letter, image_name)
                    image = cv.imread(image_path)
                    size_x, size_y, _ = image.shape

                    image = prepare_for_padding(image, output_size)
                    for n, i in [
                        ("", image),
                        ("-eroded", cv.erode(image, kernel)),
                        ("-dilated", cv.dilate(image, kernel)),
                        ("-right-shear", cv.warpAffine(image, M1, output_size, borderValue=WHITE)),
                        ("-left-shear", cv.warpAffine(image, M2, output_size, borderValue=WHITE)),
                    ]:
                        save_name = image_name.split('.')
                        save_name[0] += n
                        save_name = '.'.join(save_name)
                        cv.imwrite(os.path.join(output_folder, save_name), i)


if __name__ == '__main__':
    path = 'raw_data'
    output_path = 'training_input'
    pre_process(path=path, output_path=output_path)
