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

nclasses = 10
data = DenseNet3D()

#-------------------------------------------------------------------------------- 

def main (nnClassCount=nclasses):
	# "Define Architectures and run one by one"

	nnArchitectureList = [
			           		{
			           			'name': 'densenet3D_scanner_1_female', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_1_female_train_test_split.csv'
			           		},
			           		{
			           			'name': 'densenet3D_scanner_2_female', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_2_female_train_test_split.csv'
			           		},
			           		{
			           			'name': 'densenet3D_scanner_3_female', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_3_female_train_test_split.csv'
			           		},
			           		{
			           			'name': 'densenet3D_scanner_1_male', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_1_male_train_test_split.csv'
			           		},
			           		{
			           			'name': 'densenet3D_scanner_2_male', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_2_male_train_test_split.csv'
			           		},
			           		{
			           			'name': 'densenet3D_scanner_3_male', 
			           			'model' : DenseNet3D(out_number = nnClassCount), 
			           			'Path': '../processed_data/scanner_3_male_train_test_split.csv'
			           		}						
				    	]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)



def getDataPaths(path, mode):
 	data = pd.read_csv(path)
 	imgpaths = data[data[mode]]['Volume_Path'].as_matrix()
 	# imglabels = data[data[mode]]['Labels'].as_matrix() - 1.0
 	# print imglabels
 	return imgpaths

#--------------------------------------------------------------------------------   

def runTrain(nnArchitecture = None, mode = 'male', scanner=1):
	
	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime
	
	Path = nnArchitecture['Path']

	#---- Path to the directory with images
	TrainVolPaths = getDataPaths(Path, 'Training')
	ValidVolPaths = getDataPaths(Path, 'Validation')

	nnClassCount = nclasses
	
	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 4
	trMaxEpoch = 30
	
	#---- Parameters related to image transforms: size of the down-scaled image, cropped image
	imgtransResize = 82
	imgtransCrop = 64
	
	print ('Training NN architecture = ', nnArchitecture['name'])

	Trainer.train(TrainVolPaths,  ValidVolPaths, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)


#-------------------------------------------------------------------------------- 

def runTest():
	
	Path = '../processed_data/train_test_split.csv'
	TestVolPaths = getDataPaths(Path, 'Testing')
	nnClassCount = nclasses

	trBatchSize = 1
	imgtransResize = 82
	imgtransCrop = 64
	
	pathsModel = ['../models/densenet3D.csv']
	
	timestampLaunch = ''

	# nnArchitecture = DenseNet121(nnClassCount, nnIsTrained)
	print ('Testing the trained model')
	Tester.test(TestVolPaths, pathsModel, nnClassCount, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
#-------------------------------------------------------------------------------- 

if __name__ == '__main__':	
	main()
	# runTest()
