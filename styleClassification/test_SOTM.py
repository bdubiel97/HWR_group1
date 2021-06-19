# Testing the self-organizing time map

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

def classifySOTM(SOTM, data_point):
	classes = ['Archaic', 'Hasmonean', 'Herodian']
	#potenitally useful: 
	result = ""

	arc_dist, has_dist, her_dist = [], [], []

	arc_weights = SOTM[0].get_weights()
	has_weights = SOTM[1].get_weights()
	her_weights = SOTM[2].get_weights()

	for w in arc_weights:
		a = SOTM[0]._manhattan_distance(d,w)
		print("a: ", a)
		print("d: ", d)
		print("w: ", w)
		arc_dist.append(a)
	for w in has_weights:
		has_dist.append(SOTM[1]._manhattan_distance(d,w))
	for w in arc_weights:
		her_dist.append(SOTM[2]._manhattan_distance(d,w))
	print(arc_dist)
	print("min: ",min(arc_dist))
	print(has_dist)
	print("min: ",min(has_dist))
	print(her_dist)
	print("min: ",min(her_dist))
	mins = [min(arc_dist), min(has_dist), min(her_dist)]
	classification = classes[np.argmin(mins)]
	results = classification
	return result


#################################################################

if __name__ == '__main__':
	SOTM = []
	with open('sotm6x6_arc.p', 'wb') as infile:
		SOTM.append(pickle.load(infile))
	with open('sotm6x6_has.p', 'wb') as infile:
		SOTM.append(pickle.load(infile))
	with open('sotm6x6_her.p', 'wb') as infile:
		SOTM.append(pickle.load(infile))
	

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
			result.append(classifySOTM(som, im))
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
