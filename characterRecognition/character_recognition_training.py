import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout

# Change these to the split data subfolders
train_dir = "splitCharacterData/train"  # Use "monkbrill1" instead for training on full dataset
val_dir = "splitCharacterData/val"
test_dir = "splitCharacterData/test"

input_folder = "testSet"  # Change this to the folder containing the test set, with subfolders of the individual files, subfolders per line, with segmented characters
output_folder = "output"
batch_size = 10
image_size = 128
drop_hidden = 0.2  # {0.1, 0.2, 0.3, 0.4, 0.5}

Hebrew_alphabet = "אעבדגהחכךלםמןנפףקרסשתטץצויז"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_size, image_size),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_size, image_size),
    batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(image_size, image_size),
    batch_size=batch_size)

class_names = train_ds.class_names

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())
model.add(Dropout(drop_hidden))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())
model.add(Dropout(drop_hidden))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())
model.add(Dropout(drop_hidden))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())
model.add(Dropout(drop_hidden))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(drop_hidden))

model.add(Dense(27, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_ds,
    val_ds,
    epochs=10
)

model.save('trainedModels/CNNmodel')

# Plot results for training
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
plt.show()
plt.savefig('Accuracy_graph.png')
