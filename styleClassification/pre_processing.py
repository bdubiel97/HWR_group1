import os
from math import floor

import cv2 as cv
import numpy as np

path = 'raw_data'
output_path = 'training_input'

IMAGE_MORPH = False
if IMAGE_MORPH:
    from imagemorph import elastic_morphing
amps = [x / 10 for x in range(10)]
sigmas = range(10)


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
                              cv.BORDER_CONSTANT, value=[255, 255, 255])
    return image


def pad_and_save(image, output_size, output_folder, image_name):
    size_x, size_y, _ = image.shape
    # Resize if necessary
    if size_x > output_size[0] or size_y > output_size[1]:
        image = cv.resize(image, (floor(size_y / 2), floor(size_x / 2)))

    image = pad(image=image, output_size=output_size)
    cv.imwrite(os.path.join(output_folder, image_name), image)


def pre_process(path, output_path, output_size=None):
    if not output_size:
        output_size = [128, 128]

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

                    if IMAGE_MORPH:
                        for amp, sigma in [(a, s) for a in amps for s in sigmas]:
                            # apply random elastic morphing
                            image_m = elastic_morphing(np.array(image.convert("RGB")), amp, sigma, size_y, size_x)

                            name = image_name.split('.')
                            name[0] += '-a{}s{}'.format(amp, sigma)
                            name = '.'.join(name)
                            pad_and_save(image_m, output_size, output_folder, name)
                    else:
                        pad_and_save(image, output_size, output_folder, image_name)


if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pre_process(path=path, output_path=output_path)
