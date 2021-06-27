'''
File for testing the Self-organizing Map
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
from os import path, listdir, makedirs

input_folder = path.join('../../../Segmentation', 'output')
output_folder = "style_results"
img_size = 64

def load_data():
	'''
	This function loads the data from "characters_for_style_classification_morph" 
	per class and creates 6 arrays, data and labels for each class
	'''
	archaic = []
	hasmonean = []
	herodian = []
	arc = [f for f in iglob('characters_for_style_classification_morph/Archaic/**/*.jpg', recursive=True) if os.path.isfile(f)]
	has = [f for f in iglob('characters_for_style_classification_morph/Hasmonean/**/*.jpg', recursive=True) if os.path.isfile(f)]
	her = [f for f in iglob('characters_for_style_classification_morph/Herodian/**/*.jpg', recursive=True) if os.path.isfile(f)]

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


def classifySOM(som, data_point, winmap):
	'''
	This function classifies the testing data per character.
	'''
	default_class = np.sum(list(winmap.values())).most_common()[0][0]
	result = ""

	win_position = som.winner(data_point)
	if win_position in winmap:
		result = winmap[win_position].most_common()[0][0]
	else:
		result = default_class
	return result


if __name__ == '__main__':
	# load the saved SOM
	with open('som.p', 'rb') as infile:
		som = pickle.load(infile)
	
	# load the training data to create the win map based on the data
	archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
	train_data = list(itertools.chain(archaic,hasmonean, herodian))
	train_labels = list(itertools.chain(arc_labels, has_labels, her_labels))
	# create the win map that will be used for classification
	win_map = som.labels_map(train_data, train_labels) 

	# create the results directory if it does not exist
	if not path.exists(output_folder):
		makedirs(output_folder)

	# for each folder (document) in the output, classify the document
	for folder in listdir(input_folder):
		result = []
		files = [f for f in iglob(path.join(input_folder, folder, '*.jpg'), recursive=True) if os.path.isfile(f)]
		for file in files:
			im = imageio.imread(file)
			im = np.resize(im, (img_size, img_size, 3))
			im = im.reshape(np.prod(im.shape))
			result.append(classifySOM(som, im, win_map)) # find the classification of each character in the document
		decision = max(result, key = result.count) # determine the most frequent classification for the final decision

		# write the decision to a file with the name of the folder 
		output_path = path.join(output_folder, folder + '_style.txt')
		with open(output_path, 'w') as output:
			output.write(decision)

