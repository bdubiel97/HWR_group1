from os import listdir
from os import path

import matplotlib.pyplot as plt

from model import RecognitionModel


def plot_result(history, title="Basic Model"):
    m = min(min(history.history['accuracy']), min(history.history['val_accuracy']))
    m = (int(m * 10) - 1) / 10
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(label=title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([m, 1])
    plt.legend(loc='lower right')
    plt.show()


def train_model(input_folder, model_name):
    model = RecognitionModel(categories=len(listdir(input_folder)), save_path=model_name, model=model_name)
    model.set_input_data(input_folder)
    result = model.train(epochs=3)
    model.save_weights(model_name)
    plot_result(result, title="Style Classification using CNN")


if __name__ == '__main__':
    input_folder = 'training_input'
    model_folder = 'trained_models'
    model_name = path.join(model_folder, "new", "model")
    train_model(input_folder, model_folder)
