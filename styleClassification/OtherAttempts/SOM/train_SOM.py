'''
File for training the Self-organizing Map
'''
import numpy as np
import imageio
from glob import glob, iglob
import os
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import itertools
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

img_size = 64

def load_data():
	'''
	This function loads the data from "characters_for_style_classification_morph" 
	per class and creates 6 arrays, data and labels for each class
	'''
	archaic = []
	hasmonean = []
	herodian = []
	arc = [f for f in iglob('characters_for_style_classification_arc/Archaic/**/*.jpg', recursive=True) if os.path.isfile(f)]
	has = [f for f in iglob('characters_for_style_classification_arc/Hasmonean/**/*.jpg', recursive=True) if os.path.isfile(f)]
	her = [f for f in iglob('characters_for_style_classification_arc/Herodian/**/*.jpg', recursive=True) if os.path.isfile(f)]

	for image in arc:
		im = imageio.imread(image)
		im = np.resize(im, (img_size, img_size, 3))
		im = im.reshape(np.prod(im.shape))
		archaic.append(im)
	for image in has:
		im = imageio.imread(image)
		im = np.resize(im, (img_size, img_size, 3))
		im = im.reshape(np.prod(im.shape))
		hasmonean.append(im)
	for image in her:
		im = imageio.imread(image)
		im = np.resize(im, (img_size, img_size, 3))
		im = im.reshape(np.prod(im.shape))
		herodian.append(im)

	arc_labels = ["Archaic"] * len(archaic)
	has_labels = ["Hasmonean"] * len(hasmonean)
	her_labels = ["Herodian"] * len(herodian)
	return archaic, hasmonean, herodian, arc_labels, has_labels, her_labels

def classifySOM(som, data, x_data, y_data):
	'''
	This function classifies the testing data per character.
	'''
	winmap = som.labels_map(x_data, y_data)
	default_class = np.sum(list(winmap.values())).most_common()[0][0]
	result = []

	for d in data: 
		win_position = som.winner(d)
		if win_position in winmap:
			result.append(winmap[win_position].most_common()[0][0])
		else:
			result.append(default_class)
	return result

#################################################################
if __name__ == "__main__":
	# load the data
	archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
	print(len(archaic), len(hasmonean), len(herodian))

	# split and concatinate the data
	data = list(itertools.chain(archaic,hasmonean, herodian))
	labels = list(itertools.chain(arc_labels, has_labels, her_labels))
	x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels)

	# create the self-organizing map
	som = MiniSom(6, 6, len(data[0]), learning_rate=0.5, sigma=3)
	som.train_random(x_train, 100) # train
	som.pca_weights_init(x_train)
	print("SOM has been trained.")

	with open('som.p', 'wb') as outfile:
		pickle.dump(som, outfile)
	print("SOM has been saved.")

	print("Classifying...")
	classifications = classifySOM(som, x_test, x_train, y_train) # classify the testing data

	# printing the classification results
	print("num archaic: ", classifications.count("Archaic"), "Y TEST: ", y_test.count("Archaic"))
	print("num hasmonean: ", classifications.count("Hasmonean"), "Y TEST: ", y_test.count("Hasmonean"))
	print("num herodian: ", classifications.count("Herodian"), "Y TEST: ", y_test.count("Herodian"))
	print(accuracy_score(y_test, classifications))
	print(classification_report(y_test, classifications))
