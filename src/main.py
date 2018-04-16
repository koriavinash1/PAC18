import os
import numpy as np
import time
import sys
import pandas as pd

from DensenetModels import DenseNet3D
from Trainer_Tester import Trainer

import torch
from torch.autograd import Variable
from Trainer_Tester import Tester
# from Inference import Inference

Trainer = Trainer()
Tester  = Tester()

nclasses = 2
data = DenseNet3D(2)

#-------------------------------------------------------------------------------- 

def main (nnClassCount=nclasses):
	nnArchitectureList = [{'name': 'densenet3D', 'model' : DenseNet3D(num_classes = nnClassCount)}]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)
  
def getDataPaths(path, mode):
 	data = pd.read_csv(path)
 	imgpaths = data[data[mode]]['Volume Path'].as_matrix()
 	imglabels = data[data[mode]]['Labels'].as_matrix() - 1.0
 	# print imglabels
 	return imgpaths, imglabels

#--------------------------------------------------------------------------------   

def runTrain(nnArchitecture = None):
	
	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime
	
	Path = '../processed_data/train_test_split.csv'
	#---- Path to the directory with images
	TrainVolPaths, TrainLabels = getDataPaths(Path, 'Training')
	ValidVolPaths, ValidLabels = getDataPaths(Path, 'Validation')

	nnClassCount = nclasses
	
	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 2
	trMaxEpoch = 30
	
	#---- Parameters related to image transforms: size of the down-scaled image, cropped image
	imgtransResize = 82
	imgtransCrop = 64
	
	print ('Training NN architecture = ', nnArchitecture['name'])

	Trainer.train(TrainVolPaths, TrainLabels,  ValidVolPaths, ValidLabels, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)


#-------------------------------------------------------------------------------- 

def runTest():
	
	Path = '../processed_data/train_test_split.csv'
	TestVolPaths, TestLabels = getDataPaths(Path, 'Testing')
	nnClassCount = nclasses

	trBatchSize = 2
	imgtransResize = 82
	imgtransCrop = 64
	
	pathsModel = ['../models/modeldensenet3D.csv']
	
	timestampLaunch = ''

	# nnArchitecture = DenseNet121(nnClassCount, nnIsTrained)
	print ('Testing the trained model')
	Tester.test(TestVolPaths, TestLabels, pathsModel, nnClassCount, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
#-------------------------------------------------------------------------------- 

if __name__ == '__main__':	
	main()
	# runTest()
