from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 123

MODEL_LARGE = 'large'


class RecognitionModel(Sequential):
    def __init__(self, categories, image_size=None, save_path=None, optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy', metrics=None, model=None):
        self.training, self.validation = None, None
        self.checkpoint = None
        if not metrics:
            metrics = ['accuracy']

        if not image_size:
            image_size = [128, 128]
        self.image_size = image_size
        if save_path:
            self.checkpoint = ModelCheckpoint(filepath=save_path, save_weights_only=True, verbose=1)
        super(RecognitionModel, self).__init__()
        if model == MODEL_LARGE:
            self.set_model_large(categories=categories)
        else:
            self.set_model(categories=categories)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def set_input_data(self, folder_path):
        g = ImageDataGenerator(validation_split=0.2)
        # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # # Normalize pixel values to be between 0 and 1
        # training, test = train_images / 255.0, test_images / 255.0

        self.training = g.flow_from_directory(
            directory=folder_path,
            subset='training',
            seed=SEED
        )
        self.validation = g.flow_from_directory(
            directory=folder_path,
            subset='validation',
            seed=SEED
        )

    def set_model(self, categories):
        shape = (*self.image_size, 3)
        self.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=shape))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Flatten())

        self.add(Dense(64, activation="relu"))
        self.add(Dense(128, activation="relu"))

        self.add(Dense(27, activation="softmax"))
        self.add(Dense(categories, activation="softmax"))

        self.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def set_model_large(self, categories):
        shape = (*self.image_size, 3)
        self.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=shape))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        self.add(Flatten())

        self.add(Dense(128, activation="relu"))
        self.add(Dense(128, activation="relu"))

        self.add(Dense(27, activation="softmax"))
        self.add(Dense(categories, activation="softmax"))

        self.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=3):
        if not (self.training or self.validation or self.trainable):
            print("Data not initialized, use RecognitionModel.set_input_data(folder_part)")
            return None
        return super(RecognitionModel, self).fit(self.training, validation_data=self.validation,
                                                 epochs=epochs, callbacks=[self.checkpoint])
