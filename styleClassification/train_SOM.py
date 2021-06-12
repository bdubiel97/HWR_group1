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

def load_data():
	archaic = []
	hasmonean = []
	herodian = []
	arc = [f for f in iglob('characters_for_style_classification/Archaic/**/*.jpg', recursive=True) if os.path.isfile(f)]
	has = [f for f in iglob('characters_for_style_classification/Hasmonean/**/*.jpg', recursive=True) if os.path.isfile(f)]
	her = [f for f in iglob('characters_for_style_classification/Herodian/**/*.jpg', recursive=True) if os.path.isfile(f)]

	for image in arc:
		im = imageio.imread(image)
		im = np.resize(im, (64, 64, 3))
		im = im.reshape(np.prod(im.shape))
		archaic.append(im)
	for image in has:
		im = imageio.imread(image)
		im = np.resize(im, (64, 64, 3))
		im = im.reshape(np.prod(im.shape))
		hasmonean.append(im)
	for image in her:
		im = imageio.imread(image)
		im = np.resize(im, (64, 64, 3))
		im = im.reshape(np.prod(im.shape))
		herodian.append(im)

	arc_labels = ["Archaic"] * len(archaic)
	has_labels = ["Hasmonean"] * len(hasmonean)
	her_labels = ["Herodian"] * len(herodian)
	return archaic, hasmonean, herodian, arc_labels, has_labels, her_labels

def classifySOM(som, data, x_data, y_data):
	winmap = som.labels_map(x_data, y_data) 
	default_class = np.sum(list(winmap.values())).most_common()[0][0]
	print(default_class)
	result = []

	for d in data: 
		win_position = som.winner(d)
		if win_position in winmap:
			result.append(winmap[win_position].most_common()[0][0])
		else:
			result.append(default_class)
	return result

#################################################################

archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
print(len(archaic), len(hasmonean), len(herodian))
# concatinate data
data = list(itertools.chain(archaic,hasmonean, herodian))
labels = list(itertools.chain(arc_labels, has_labels, her_labels))
print(len(data))

x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels)
print("num archaic: ", y_train.count("Archaic"))
print("num hasmonean: ", y_train.count("Hasmonean"))
print("num herodian: ", y_train.count("Herodian"))

som = MiniSom(25, 25, len(data[0]), learning_rate=0.5, sigma=3)
som.train_random(x_train, 100)
som.pca_weights_init(x_train)
with open('som15x15.p', 'wb') as outfile:
	pickle.dump(som, outfile)

classifications = classifySOM(som, x_test, x_train, y_train)
print("num archaic: ", classifications.count("Archaic"), "Y TEST: ", y_test.count("Archaic"))
print("num hasmonean: ", classifications.count("Hasmonean"), "Y TEST: ", y_test.count("Hasmonean"))
print("num herodian: ", classifications.count("Herodian"), "Y TEST: ", y_test.count("Herodian"))
print(accuracy_score(y_test, classifications))
print(classification_report(y_test, classifications))

""" 
accuracies: 
6x6 = 0.35700934579439253
10x10 = 0.35677570093457944
15x15 = 0.3733644859813084
20x20 = 
25x25 = 0.3616822429906542
"""
