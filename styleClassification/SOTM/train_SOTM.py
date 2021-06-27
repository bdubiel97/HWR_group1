'''
File for training the Self-organizing Time Map
'''
import numpy as np
import imageio
from glob import glob, iglob
import os
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from SOM.som import SOM, man_dist_pbc
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import expand_dims, nn
from os import path, listdir, makedirs

img_size = 128
classes = {0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}

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

	arc_labels = [0] * len(archaic)
	has_labels = [1] * len(hasmonean)
	her_labels = [2] * len(herodian)
	return archaic, hasmonean, herodian, arc_labels, has_labels, her_labels

def classify(test_data, SOTM):
	'''
	This function classifies the testing data per character.
	'''
	results = []

	for d in test_data:
		arc_dist = np.min(np.sum((SOTM[0].map - d) ** 2, axis=2))
		has_dist = np.min(np.sum((SOTM[1].map - d) ** 2, axis=2)) 
		her_dist = np.min(np.sum((SOTM[2].map - d) ** 2, axis=2)) 
		mins = [arc_dist, has_dist, her_dist]
		classification = np.argmin(mins)
		results.append(classification)
	return results

#################################################################
if __name__ == "__main__":
	# load the data
	archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
	print(len(archaic), len(hasmonean), len(herodian))
	# turn into an array
	archaic = np.asarray(archaic)
	hasmonean = np.asarray(hasmonean)
	herodian = np.asarray(herodian)
	# split and concatinate the data
	arc_Xtrain, arc_Xtest, arc_Ytrain, arc_Ytest = train_test_split(archaic, arc_labels, stratify=arc_labels)
	has_Xtrain, has_Xtest, has_Ytrain, has_Ytest = train_test_split(hasmonean, has_labels, stratify=has_labels)
	her_Xtrain, her_Xtest, her_Ytrain, her_Ytest = train_test_split(herodian, her_labels, stratify=her_labels)
	x_test = np.concatenate((arc_Xtest, has_Xtest, her_Xtest))
	y_test = np.concatenate((arc_Ytest, has_Ytest, her_Ytest))

	# create the self-organizing time map
	SOTM = []
	arc_som = SOM(20,20) 
	arc_som.fit(arc_Xtrain, 1000, save_e = True, interval = 100)
	SOTM.append(arc_som) # add the SOM for the archaic time period
	print("Archaic SOM is trained")

	has_som = SOM(20,20)
	has_som.map = arc_som.map
	has_som.epoch = 0
	has_som.fit(has_Xtrain, 1000, save_e = True, interval = 100)
	SOTM.append(has_som) # add the SOM for the hasmonean time period
	print("Hasmonean SOM is trained")

	her_som = SOM(20,20)
	her_som.map = has_som.map
	her_som.epoch = 0
	her_som.fit(her_Xtrain, 1000, save_e = True, interval = 100)
	SOTM.append(her_som) # add the SOM for the herodian time period
	print("Herodian SOM is trained")
	
	# save the sub-maps of the SOTM
	arc_som.save("SOTM_arc.p")
	has_som.save("SOTM_has.p")
	her_som.save("SOTM_her.p")
	print("SOTM has been saved.")

	print("Classifying...")
	classifications = classify(x_test, SOTM) # classify the testing data

	# printing the classification results
	print("num archaic: ", classifications.count(0), "Y TEST: ", np.count_nonzero(y_test == 0))
	print("num hasmonean: ", classifications.count(1), "Y TEST: ", np.count_nonzero(y_test == 1))
	print("num herodian: ", classifications.count(2), "Y TEST: ", np.count_nonzero(y_test == 2))
	print(accuracy_score(y_test, classifications))
	print(classification_report(y_test, classifications))
