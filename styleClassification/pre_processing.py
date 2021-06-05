import os
from math import floor

from PIL import Image

path = 'raw_data'
output_path = 'training_input'
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_size = [128, 128]
for style in os.listdir(path):
    output_folder = os.path.join(output_path, style)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for letter in os.listdir(os.path.join(path, style)):
        for image_name in os.listdir(os.path.join(path, style, letter)):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(path, style, letter, image_name)
                image = Image.open(image_path)
                size_x, size_y = image.size

                # Resize if necessary
                if any([True for x in image.size if x > 128]):
                    image = image.resize((floor(size_x / 2), floor(size_y / 2)))
                    size_x, size_y = image.size
                # Calculate center of gravity
                pixels = list(image.convert('1').getdata())
                row_indices = [x for x in range(size_x) for y in range(size_y) if pixels[x * size_y + y] != 0]
                row_indices.sort()
                mean_x = row_indices[floor(len(row_indices) / 2)]
                col_indices = [y for x in range(size_x) for y in range(size_y) if pixels[x * size_y + y] != 0]
                col_indices.sort()
                mean_y = col_indices[floor(len(col_indices) / 2)]

                # Pad image to output_size
                pos_x = int(max(0, min(output_size[0] - size_x, output_size[0] / 2 - mean_x)))
                pos_y = int(max(0, min(output_size[1] - size_y, output_size[1] / 2 - mean_y)))

                new_image = Image.new(mode='RGB', size=output_size, color='white')
                new_image.paste(image, (pos_x, pos_y))

                # Save image in training folder
                new_image.save(os.path.join(output_folder, image_name))
