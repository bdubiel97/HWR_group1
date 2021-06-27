'''
File for testing the Self-organizing Time Map
'''
import numpy as np
import imageio
from glob import glob, iglob
import os
import itertools
from sklearn.model_selection import train_test_split
from SOM.som import SOM
from os import path, listdir, makedirs

input_folder = path.join('../../Segmentation', 'output')
output_folder = "../../results"
img_size = 128
classes = {0: 'Archaic', 1: 'Hasmonean', 2: 'Herodian'}

def classify(SOTM, d):
	''' 
	This function finds the smallest distance from each map to the new data point and returns the classification based
	on the smallest of these values
	'''
	arc_dist = np.min(np.sum((SOTM[0].map - d) ** 2, axis=2))
	has_dist = np.min(np.sum((SOTM[1].map - d) ** 2, axis=2)) 
	her_dist = np.min(np.sum((SOTM[2].map - d) ** 2, axis=2)) 
	mins = [arc_dist, has_dist, her_dist]
	classification = np.argmin(mins)
	result = classes[classification]

	return result

#################################################################
if __name__ == "__main__":
	
	# load the saved sub-maps for the SOTM
	SOTM = []
	arc_som = SOM(15,15) 
	arc_som.load("SOTM_arc.p")
	SOTM.append(arc_som)
			
	has_som = SOM(15,15)
	has_som.load("SOTM_has.p")
	SOTM.append(has_som)
			
	her_som = SOM(15,15) 
	her_som.load("SOTM_her.p")
	SOTM.append(her_som)
			
	print("The SOTM has been loaded.")
	print("Classifying...")

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
			result.append(classify(SOTM, im)) # find the classification of each character in the document
		decision = max(result, key = result.count) # determine the most frequent classification for the final decision

		# write the decision to a file with the name of the folder 
		output_path = path.join(output_folder, folder + '_style.txt')
		with open(output_path, 'w') as output:
			output.write(decision)
