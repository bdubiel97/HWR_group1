#Uses the original provided input monkbrill files to generate more, augmented data. 
import tensorflow as tf
import numpy as np
import cv2 as cv
import os, sys
from PIL import Image
import argparse
import skimage
from skimage import io

#Set this to the data directory with subfolders per character. The augmented data will be added into it. 
imDir = "monkbrill3"
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
		#im.resize(imSize, imSize)

		imErode = cv.erode(im, kernel)
		imDilate = cv.dilate(im, kernel)
		  ## Use these to see the effect of erosion and dilation imediately
		#cv.imshow('Erosion', imErode)
		#cv.imshow('dilation', imDilate)
		#cv.waitKey(0)
		
		#Shearing is currently broken so I'm just trying the other things
		#imSheared = cv.warpPerspective(im,M,int(imSize))

		#(imDir + path + "Erode.jpg")
		#imErode.save(imDir + path + file + "Erode.jpg")
		#imDilate.save(imDir + path + file + "Dilate.jpg")
		cv.imwrite(path + file + "Erode.jpg", imErode)
		cv.imwrite(path + file + "Dilate.jpg", imDilate)

		#im.save(imDir + file + ".PNG", "PNG")
		#imErode.save(imDir + path + file + "Erode.jpg")
		#imDilate.save(imDir + path + file + "Dilate.jpg")
		#im.save(imDir + dirExtension[i] + file + "Shear.PNG", "PNG")









