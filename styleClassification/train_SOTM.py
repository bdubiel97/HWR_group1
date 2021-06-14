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

def classifySOTM(SOTM, data):
	classes = ['Archaic', 'Hasmonean', 'Herodian']
	#potenitally useful: 
	results = []
	for d in data: 
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
		results.append(classification)
	return results


#################################################################

archaic, hasmonean, herodian, arc_labels, has_labels, her_labels = load_data()
print(len(archaic), len(hasmonean), len(herodian))
# concatinate data
arc_Xtrain, arc_Xtest, arc_Ytrain, arc_Ytest = train_test_split(archaic, arc_labels, stratify=arc_labels)
has_Xtrain, has_Xtest, has_Ytrain, has_Ytest = train_test_split(hasmonean, has_labels, stratify=has_labels)
her_Xtrain, her_Xtest, her_Ytrain, her_Ytest = train_test_split(herodian, her_labels, stratify=her_labels)

x_test = list(itertools.chain(arc_Xtest, has_Xtest, her_Xtest))
y_test = list(itertools.chain(arc_Ytest, has_Ytest, her_Ytest))
print(len(x_test))
print(len(y_test))

SOTM = []
arc_som = MiniSom(6, 6, len(arc_Xtrain[0]), learning_rate=0.5, sigma=3)
arc_som.train_random(arc_Xtrain, 1)
arc_som.pca_weights_init(arc_Xtrain)
SOTM.append(arc_som)
print("archaic is done training")
has_som = arc_som
has_som.train_random(has_Xtrain, 1)
has_som.pca_weights_init(has_Xtrain)
SOTM.append(has_som)
print("hasmonean is done training")
her_som = has_som
her_som.train_random(her_Xtrain, 1)
her_som.pca_weights_init(her_Xtrain)
SOTM.append(her_som)
print("herodian is done training")

with open('som6x6_arc.p', 'wb') as outfile:
	pickle.dump(arc_som, outfile)
with open('som6x6_has.p', 'wb') as outfile:
	pickle.dump(has_som, outfile)
with open('som6x6_her.p', 'wb') as outfile:
	pickle.dump(her_som, outfile)

classifications = classifySOTM(SOTM, x_test)
print("num archaic: ", classifications.count("Archaic"), "Y TEST: ", y_test.count("Archaic"))
print("num hasmonean: ", classifications.count("Hasmonean"), "Y TEST: ", y_test.count("Hasmonean"))
print("num herodian: ", classifications.count("Herodian"), "Y TEST: ", y_test.count("Herodian"))
print(accuracy_score(y_test, classifications))
print(classification_report(y_test, classifications))



""" 
accuracies: 
10x10 = 0.35677570093457944
15x15 = 0.3733644859813084
"""
