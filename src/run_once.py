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


imgtransCrop   = 64
imgtransResize = 82
trBatchSize    =4
transformList = {}
transformList['MinMax'] = True
transformList['ZScore'] = True
transformList['Resize'] = imgtransResize
transformList['RandomResizedCrop'] = imgtransCrop
transformList['Sequence'] = True

Path='../processed_data/scanner_1.csv'

def getDataPaths(path, mode):
 	data = pd.read_csv(path)
 	imgpaths = data[data[mode]]['Volume_Path'].as_matrix()
 	# imglabels = data[data[mode]]['Labels'].as_matrix() - 1.0
 	# print imglabels
 	return imgpaths


TrainVolPaths = getDataPaths(Path, 'Training')
ValidVolPaths = getDataPaths(Path, 'Validation')
#-------------------- SETTINGS: DATASET BUILDERS
datasetTrain = DatasetGenerator(imgs = TrainVolPaths, transform=transformList)
datasetVal =   DatasetGenerator(imgs =ValidVolPaths, transform=transformList)

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=8, pin_memory=False)
a,b,c,d,_= next(iter(dataLoaderTrain ))

model = DenseNet3D()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


a= a.to(device)
b =b.type(torch.LongTensor)

b=b.to(device)
c= c.to(device)
d= d.to(device)
model=model.cuda()
outs = model(a,c,d)

criterion=nn.CrossEntropyLoss()
# err= criterion(outs,b)