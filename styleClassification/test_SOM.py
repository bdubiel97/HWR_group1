# Testing the Self-organizing map

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
from os import path, listdir

input_folder = path.join('images_for_style', 'output')

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


def classifySOM(som, data_point, winmap):
	default_class = np.sum(list(winmap.values())).most_common()[0][0]
	result = ""

	win_position = som.winner(data_point)
	if win_position in winmap:
		result = winmap[win_position].most_common()[0][0]
	else:
		result = default_class
	return result


if __name__ == '__main__':
	with open('som6x6.p', 'rb') as infile:
		som = pickle.load(infile)
	
	archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
	train_data = list(itertools.chain(archaic,hasmonean, herodian))
	train_labels = list(itertools.chain(arc_labels, has_labels, her_labels))
	win_map = som.labels_map(train_data, train_labels)


	for folder in listdir(input_folder):
		print(folder)
		result = []
		count = 0
		files = [f for f in iglob(path.join(input_folder, folder, '*.jpg'), recursive=True) if os.path.isfile(f)]
		print(len(files))
		for file in files:
			im = imageio.imread(file)
			im = np.resize(im, (64, 64, 3))
			im = im.reshape(np.prod(im.shape))
			result.append(classifySOM(som, im, win_map))
			count += 1
		print("archaic: ", result.count("Archaic"))
		print("hasmonean: ", result.count("Hasmonean"))
		print("herodian: ", result.count("Herodian"))
		decision = max(result, key = result.count)
		print("results: ", decision)
		print(count)
		
		output_path = folder + '_style.txt'
		with open(output_path, 'w') as output:
			output.write(decision)

