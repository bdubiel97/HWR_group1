from os import listdir, path, makedirs
from re import match

import cv2 as cv

pattern = "^.*-{}=([0-9]+).*$"
name_pattern = "^(.*)-x=[0-9]+-y=[0-9]+-h=[0-9]+.jpg$"


class Im:
    def __init__(self, name, image, pos_x, pos_y, h):
        self.name = name
        self.image = image
        self.x = pos_x
        self.y = pos_y
        self.h = h


def load_images(input_image, output_path):
    images = []

    if not path.exists(output_path):
        makedirs(output_path)
    for letter in listdir(input_image):
        if letter.endswith('.jpg'):
            # Read image
            try:
                image_name = match(name_pattern, letter).groups()[0]
                image_path = path.join(input_image, letter)
                image = cv.imread(image_path)
                info = []
                for t in ['x', 'y', 'h']:
                    info.append(int(match(pattern.format(t), letter).groups()[0]))
                images.append(Im(image_name, image, *info))
            except ValueError:
                pass
    return images


def save_images(images: [[Im]], output_path: str):
    for row, row_index in zip(images, range(len(images))):
        output_folder = path.join(output_path, str(row_index))
        if not path.exists(output_folder):
            makedirs(output_folder)
        for image, index in zip(row, range(len(row))):
            image_path = path.join(output_path, str(row_index), str(index) + '.jpg')
            print("Saving image:", image_path)
            cv.imwrite(image_path, image.image)


def sort_images(images):
    images.sort(key=lambda l: l.y)
    new_images = []
    heights = [i.h for i in images]
    line_height = sum(heights)/len(heights) * 1.2
    while len(images):
        new_images.append([i for i in images if i.y < images[0].y + line_height])
        new_images[-1].sort(key=lambda l: l.x, reverse=True)
        [images.remove(i) for i in new_images[-1]]
    return new_images


def from_segmentation_to_character_recognition(input_path, output_path):
    for input_image in listdir(input_path):
        images = load_images(input_image=path.join(input_path, input_image), output_path=output_path)
        print(len(images))
        images = sort_images(images)
        print(len(images))
        save_images(images, path.join(output_path, input_image))


if __name__ == '__main__':
    from_segmentation_to_character_recognition(input_path="output", output_path="sorted_output")
