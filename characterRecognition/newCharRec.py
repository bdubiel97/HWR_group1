import tensorflow as tf
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


data_dir = "monkbrill2" #Change this to the name of the folder with augmented/aggregated data
test_dir =  "monkbrill2"
batch_size = 10
image_size = 64


Hebrew_alphabet = ['א','ע','ב','ד','ג','ה','ח','כ','ך','ל','ם','מ','ן','נ','פ','ף','ק','ר','ס','ש','ת','ט','ץ','צ','ו','י','ז'] 
print(Hebrew_alphabet)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='categorical',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(image_size, image_size),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(image_size, image_size),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  #labels='inferred',
  #label_mode='categorical',
  #subset="validation",
  #seed=123,
  image_size=(image_size, image_size),
  batch_size=batch_size)


class_names = train_ds.class_names
#print(class_names)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(class_names)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#print(integer_encoded)
onehot_encoded_classes = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded_classes)
inverted = label_encoder.inverse_transform([argmax(onehot_encoded_classes[0, :])])
print(inverted)



#train_ds = to_categorical(train_ds,27)
#val_ds = to_categorical(val_ds,27)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_size,image_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(27,activation ="softmax"))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)

model.summary()


history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1 
)

predictions = model.predict(test_ds)
#print(predictions[0][0])
output_class = []
thisIt = []
for i in range(0,len(test_ds)):
  #thisIt.append(label_encoder.inverse_transform([argmax(predictions[i, :])])[0])
  output_class.append(argmax(predictions[i, :]))
  #print(output_class)
print(output_class)

letters=[]
for i in range(len(output_class)):
  x = output_class[i]
  print(x)
  y = Hebrew_alphabet[x]
  print(y)
  letters.append(y)
  print(letters[i])
print(letters)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

val_loss, val_acc = model.evaluate(val_ds, verbose=2)
plt.show()
plt.savefig('Accuracy_graph.png')

'''
categorisedCharacters = open(r"characters.txt", "w")
embedFile.write(str(len(word2idx)) + " 100\n")

with open("accuracies.txt", "a+") as file:
    file.write("%s,%s\n" %(change, ",".join(map(str,accuracies))))
print("Written to file.")


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
'''




