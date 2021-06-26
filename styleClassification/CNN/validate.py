from os import path, listdir

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims, nn

from styleClassification.CNN.model import RecognitionModel


def load_model(model_name, categories):
    model = RecognitionModel(categories=categories)
    model.load_weights(model_name)
    return model


def validate_image(model, file):
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
    return c


def validate_model(model_name, class_names):
    model = load_model(model_name=model_name, categories=len(class_names))
    results = {}
    with open('results.txt', 'w') as fr:
        for folder in listdir(input_folder):
            # Aggregate results for input image per segmented character
            results[folder] = []
            for file in listdir(path.join(input_folder, folder)):
                file = path.join(input_folder, folder, file)
                results.get(folder).append(validate_image(model=model, file=file))

            # Write prediction of a single input image
            c = max(results.get(folder), key=results.get(folder).count)
            image_count = [(x, results.get(folder).count(x)) for x in set(results.get(folder))]
            print(', '.join(["{}: {}".format(p, n) for p, n in image_count]))
            s = "Input image '{}' has been classified as '{}'".format(folder, c)
            print(s)
            fr.write(s + '\n')


if __name__ == '__main__':
    input_folder = path.join('..', '..', 'Segmentation', 'output')
    model_folder = 'trained_models'
    model_name = path.join(model_folder, "plus_e_d_shear", "model")

    class_names = listdir('training_input')

    validate_model(model_name=model_name, class_names=class_names)
