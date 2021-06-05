from model import RecognitionModel
from os import listdir
from os import path

input_folder = 'training_input'
model_folder = 'trained_models'
model_name = path.join(model_folder, "model")

if __name__ == '__main__':
    model = RecognitionModel(categories=len(listdir(input_folder)), save_path=model_name)
    model.set_input_data(input_folder)
    model.train()
    model.save_weights(model_name)
