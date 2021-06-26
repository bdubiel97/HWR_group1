import tensorflow as tf
import keras
import glob
import time
from numpy import argmax
from os import path, listdir, makedirs
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims, nn

results = []
image_size = 128
input_folder = "testSet"  # Change this to the folder containing the test set, with subfolders of the individual files, subfolders per line, with segmented characters
output_folder = "output"

Hebrew_alphabet = "אעבדגהחכךלםמןנפףקרסשתטץצויז"

#load the pretrained model
trained_model = keras.models.load_model('trainedModels/CNNmodel')

# Write unlabeled testig data to files with names of the input image

for input_file in listdir(input_folder):
    if not path.exists(output_folder):
        makedirs(output_folder)
    with open(path.join(output_folder, input_file + '_characters.txt'), 'w') as fr:
        row_path = path.join(input_folder, input_file)
        for row in sorted(listdir(row_path)):
            results = []
            letter_path = path.join(row_path, row)
            print(letter_path)
            time.sleep(2)
            for letter in sorted(listdir(letter_path)):
                if letter.endswith('.jpg'):
                    # Open images and predict letter
                    file = path.join(letter_path, letter)
                    img = load_img(
                        file, target_size=(image_size, image_size)
                    )
                    print(file)
                    img_array = img_to_array(img)
                    img_array = expand_dims(img_array, 0)  # Create a batch

                    predictions = trained_model.predict(img_array)
                    score = argmax(predictions[0, :])

                    results.append(score)
            letters = [Hebrew_alphabet[results[i]] for i in range(len(results))]
            fr.write("".join(letters) + '\n')

