from os import path, listdir

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims, nn

from model import RecognitionModel

input_folder = path.join('..', 'Segmentation', 'output')
model_folder = 'trained_models'
model_name = path.join(model_folder, "model")

class_names = listdir('training_input')

if __name__ == '__main__':
    model = RecognitionModel(categories=3)
    model.load_weights(model_name)

    results = {}

    with open('results.txt', 'w') as fr:
        for folder in listdir(input_folder):
            results[folder] = []
            for file in listdir(path.join(input_folder, folder)):
                file = path.join(input_folder, folder, file)
                img = load_img(
                    file, target_size=(128, 128)
                )
                img_array = img_to_array(img)
                img_array = expand_dims(img_array, 0)  # Create a batch

                predictions = model.predict(img_array)
                score = nn.softmax(predictions[0])

                c = class_names[np.argmax(score)]
                # print(
                #     "This image most likely belongs to {} with a {:.2f} percent confidence."
                #         .format(c, 100 * np.max(score))
                # )
                results.get(folder).append(c)

            c = max(results.get(folder), key=results.get(folder).count)
            s = "Input image '{}' has been classified as '{}'".format(folder, c)
            print(s)
            fr.write(s+'\n')
