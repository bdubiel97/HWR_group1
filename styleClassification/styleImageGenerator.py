#Uses the original provided input monkbrill files to generate more, augmented data. 
import tensorflow as tf
import numpy as np
import cv2 as cv
import os, sys
from PIL import Image
import argparse
import skimage
from skimage import io
from imagemorph import elastic_morphing 

#Set this to the data directory with subfolders per character. The augmented data will be added into it. 
imDir = "characters_for_style_classification/Archaic"
amp, sigma = 0.9, 9

folders = os.listdir(imDir)
if '.DS_Store' in folders:
	folders.remove('.DS_Store')

for dirExtension in folders:
	path = imDir + "/" + dirExtension + "/"
	for file in os.listdir(path):
		im = cv.imread(path + file)

		h, w, _ = im.shape

		#This if-statement is required to avoid messed up images and warnings
		if w > 27: 
			imMorph = elastic_morphing(im, amp, sigma, h, w)
			#Write all images to the folder
			cv.imwrite(path + file + "Morph.jpg", imMorph)
