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
#imSize = 64

kernel = np.ones((3,3), np.uint8)
M = np.float32([[1, 0.2, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])

folders = os.listdir(imDir)
if '.DS_Store' in folders:
	folders.remove('.DS_Store')

for dirExtension in folders:
	path = imDir + "/" + dirExtension + "/"
	print(path)
	for file in os.listdir(path):
		im = cv.imread(path + file)

		h, w, _ = im.shape
		#Matrix 1 for shearing
		M1 = np.float32([[1, -0.2, 0], [0, 1, 0]])
		M1[0,2] = -M1[0,1] * w/2
		M1[1,2] = -M1[1,0] * h/2
		#Matrix 2 for shearing
		M2 = np.float32([[1, 0, 0], [0.2, 1, 0]])
		M2[0,2] = -M2[0,1] * w/2
		M2[1,2] = -M2[1,0] * h/2


		#Create all new augmented versions of im
		#This if-statement is required to avoid messed up images and warnings
		if w > 27: 
			imMorph = elastic_morphing(im, amp, sigma, h, w)
		imErode = cv.erode(im, kernel)
		imDilate = cv.dilate(im, kernel)
		imShearedRight = cv.warpAffine(im, M1, (w, h))
		imShearedLeft = cv.warpAffine(im, M2, (w, h))
		


		  ## Use these to see the effect of erosion and dilation imediately
		#cv.imshow('Erosion', imErode)
		#cv.imshow('dilation', imDilate)
		#cv.waitKey(0)
		
		#Write all images to the folder
		if w > 25:
			cv.imwrite(path + file + "Morph.jpg", imMorph)
		cv.imwrite(path + file + "Erode.jpg", imErode)
		cv.imwrite(path + file + "Dilate.jpg", imDilate)
		cv.imwrite(path + file + "RightShear.jpg", imShearedRight)
		cv.imwrite(path + file + "LefttShear.jpg", imShearedLeft)







