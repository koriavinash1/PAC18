import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
from torch.utils.data import DataLoader

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet3D
from DataGenerator import DatasetGenerator

import torchnet as tnt
import pandas as pd

nclasses = 2
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

#-------------------------------------------------------------------------------- 

class Trainer ():
	#---- Train the densenet network 
	#---- TrainVolPaths - path to the directory that contains images
	#---- TrainLabels - path to the file that contains image paths and label pairs (training set)
	#---- ValidVolPaths - path to the directory that contains images
	#---- ValidLabels - path to the file that contains image paths and label pairs (training set)
	#---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
	#---- nnClassCount - number of output classes 
	#---- trBatchSize - batch size
	#---- trMaxEpoch - number of epochs
	#---- transResize - size of the image to scale down to (not used in current implementation)
	#---- transCrop - size of the cropped image 
	#---- launchTimestamp - date/time, used to assign unique name for the checkpoint file

	#---- TODO:
	#---- checkpoint - if not None loads the model and continues training
	
	def train (self, TrainVolPaths, ValidVolPaths, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint):

		
		#-------------------- SETTINGS: NETWORK ARCHITECTURE
		model = nnArchitecture['model'].cuda()
		
		# model = torch.nn.DataParallel(model.cuda())
				
		#-------------------- SETTINGS: DATA TRANSFORMS
		
		transformList = {}
		transformList['MinMax'] = True
		transformList['ZScore'] = True
		transformList['Resize'] = imgtransResize
		transformList['RandomResizedCrop'] = imgtransCrop
		transformList['Sequence'] = True

		#-------------------- SETTINGS: DATASET BUILDERS
		datasetTrain = DatasetGenerator(imgs = TrainVolPaths, transform=transformList)
		datasetVal =   DatasetGenerator(imgs =ValidVolPaths, transform=transformList)
			  
		dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
		dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=False)
		
		# print len(dataLoaderTrain), len(datasetTrain)
		#-------------------- SETTINGS: OPTIMIZER & SCHEDULER
		optimizer = optim.Adam (model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5)
		scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
				
		#-------------------- SETTINGS: LOSS
		loss = torch.nn.BCELoss()
		# loss = torch.nn.CrossEntropyLoss()
		
		#---- Load checkpoint 
		if checkpoint != None:
			model = torch.load(checkpoint)

		
		#---- TRAIN THE NETWORK
		
		lossMIN = 100000
		sub = pd.DataFrame()

		timestamps = []
		archs = []
		losses = []
		accs = []
		for epochID in range (0, trMaxEpoch):
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			timestampSTART = timestampDate + '-' + timestampTime
			
			print (str(epochID)+"/" + str(trMaxEpoch) + "---")
			self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
			lossVal, losstensor, acc = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
			
			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			launchTimestamp = timestampDate + '-' + timestampTime
			
			scheduler.step(losstensor.data[0])
			
			if lossVal < lossMIN:
				lossMIN = lossVal
				timestamps.append(launchTimestamp)
				archs.append(nnArchitecture['name'])
				losses.append(lossMIN)
				accs.append(acc)
				model_name = '../models/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + ' accuracy = ' + str(acc)+ '.pth.tar'

				torch.save(model, model_name)
				print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' accuracy= ' + str(acc))
				# print confusion_meter
			else:
				print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' accuracy= ' + str(acc))
		sub['timestamp'] = timestamps
		sub['archs'] = archs
		sub['loss'] = losses
		sub['acc'] = accs

		sub.to_csv('../models/' + nnArchitecture['name'] + '.csv', index=True)
		
					 
	#-------------------------------------------------------------------------------- 

	# compute accuracy
	def accuracy(self, output, labels):
		
		pred = torch.max(output, 1)[1]
		label = torch.max(labels, 1)[1]
		# print pred, label, output, labels
		acc = torch.sum(pred == label)
		return float(acc)
	

	#--------------------------------------------------------------------------------
	def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):
		
		model.train()
		for batchID, (input, target, age, tiv, _) in tqdm(enumerate (dataLoader)):
			# print 		
			target = target.cuda()
			
			varInput = torch.autograd.Variable(input.cuda(), volatile=True)
			varAge = torch.autograd.Variable(age.cuda())  
			varTiv = torch.autograd.Variable(tiv.cuda())  
			varTarget = torch.autograd.Variable(target.cuda())

			varOutput = model(varInput, varAge, varTiv)

			# print varInput.size(), varOutput.size(), target.size()
			# varOutput = torch.FloatTensor([0])

			# lossfn = loss(weights = weights)
			lossvalue = loss(varOutput, varTarget)	
			l2_reg = None
			for W in model.parameters():
			    if l2_reg is None:
			        l2_reg = W.norm(2)
			    else:
			        l2_reg = l2_reg + W.norm(2)

			lossvalue = lossvalue + l2_reg * 1e-3
			# batch_loss.backward()   
			optimizer.zero_grad()
			lossvalue.backward()
			optimizer.step()
			
	#-------------------------------------------------------------------------------- 
		
	def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):
		
		model.eval ()
		
		lossVal = 0
		lossValNorm = 0
		
		losstensorMean = 0
		confusion_meter.reset()

		acc = 0.0
		for i, (input, target, age, tiv,_) in enumerate (dataLoader):
			
			target = target.cuda()
				 
			varInput = torch.autograd.Variable(input.cuda(), volatile=True)
			varAge = torch.autograd.Variable(age.cuda(), volatile=True)  
			varTiv = torch.autograd.Variable(tiv.cuda(), volatile=True)  
			varTarget = torch.autograd.Variable(target.cuda(), volatile=True)        
			varOutput = model(varInput, varAge, varTiv)
			
			acc += self.accuracy(varOutput, varTarget)/ (len(dataLoader)*trBatchSize)
			losstensor = loss(varOutput, varTarget)
			# print varOutput, varTarget
			losstensorMean += losstensor
			# confusion_meter.add(varOutput.view(-1), varTarget.data.view(-1))
			lossVal += losstensor.data[0]
			lossValNorm += 1

		
		outLoss = lossVal / lossValNorm
		losstensorMean = losstensorMean / lossValNorm
		
		return outLoss, losstensorMean, acc
#-------------------------------------------------------------------------------- 

class Tester():
	#--------------------------------------------------------------------------------  
	#---- Test the trained network 
	#---- pathDirData - path to the directory that contains images
	#---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
	#---- pathFileVal - path to the file that contains image path and label pairs (validation set)
	#---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
	#---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
	#---- nnClassCount - number of output classes 
	#---- trBatchSize - batch size
	#---- trMaxEpoch - number of epochs
	#---- transResize - size of the image to scale down to (not used in current implementation)
	#---- transCrop - size of the cropped image 
	#---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
	#---- checkpoint - if not None loads the model and continues training

	def get_best_model_path(self, path, expert = True):
		data = pd.read_csv(path)
		acc = np.squeeze(data['acc'].as_matrix())
		timestamp = np.squeeze(data['timestamp'].as_matrix())
		arch = np.squeeze(data['archs'].as_matrix())

		index = np.where(acc == np.max(acc)) [0]
		index = index[len(index) - 1]
		try: 
			name = timestamp[index] + "-" + arch[index]
		except: 
			name = timestamp + "-" + arch
		path = '../models/model-m-' + name + '.pth.tar'
		return path
	
	def accuracy(self, output, labels):
		print (output, labels)
		acc = float(np.sum(output == labels))/len(labels)
		return acc

	def test (self, TestVolPaths, pathsModel, nnClassCount, trBatchSize, transResize, transCrop, launchTimeStamp):

		#-------------------- SETTINGS: DATA TRANSFORMS
		
		transformList = {}
		transformList['MinMax'] = True
		transformList['Resize'] = transResize

		# any one of these TenCrop and FiveCrop should be True......
		transformList['TenCrop'] = transCrop

		transformList['Sequence'] = True
		
		datasetTest = DatasetGenerator(imgs = TestVolPaths, transform=transformList)
		dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
		
		# results per image
		outGTs = torch.FloatTensor().cpu()
		outPREDs = [] 		
		image_paths = []

		
		st = time.time() 
		for i, (input, target, path) in enumerate(dataLoaderTest):
			print (path)

			target = target.cpu()
			# print input.size()
			bs, n_crops, c, h, w, l  = input.size()
			varInput = torch.autograd.Variable(input.view(-1, c, h, w, l).cpu(), volatile=True)

			max_model1 = []
			for pathModel in pathsModel:
				# best_model_path = self.get_best_model_path(pathModel, expert=False)

				best_model_path = '../models/model-m-21032018-225258-densenet3D.pth.tar'
				model = torch.load(best_model_path)

				model.eval()			
				out = model(varInput)

				_,class_associated_to_each_crop = torch.max(out,1)
				del _

				class_associated_to_each_crop = class_associated_to_each_crop.data.cpu().numpy() ### numpify
				count_for_all_classes = np.bincount(class_associated_to_each_crop)
				
				del class_associated_to_each_crop
				class_associated_to_image = np.argmax(count_for_all_classes)
				del count_for_all_classes

				max_model1.append(class_associated_to_image)
				############

			max_primary_output = np.bincount(np.array(max_model1))
			final_output = np.argmax(max_primary_output)
			# print max_model1, final_output

			outGTs = torch.cat((outGTs, target), 0)
			outPREDs.append(final_output)
			image_paths.append(path)

		outGTs = torch.max(outGTs, 1)[1]
		outGTs = outGTs.cpu().numpy()

		# per image csv....
		sub = pd.DataFrame()
		sub['path'] = image_paths
		sub['actual'] = outGTs
		sub['predicted'] = outPREDs
		
		sub.to_csv('../logs/Testing.csv', index=True)
		print ("Final Accuracy: {}".format(self.accuracy(outPREDs, outGTs)))
	
#-------------------------------------------------------------------------------- 
