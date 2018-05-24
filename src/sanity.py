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
import pandas as pd
from torch.autograd import Variable
nclasses = 2

model= DenseNet3D().cuda()
data = Variable(torch.rand(2,1,64,64,64).cuda())
age = Variable(torch.rand(2,1).cuda())
gender = Variable(torch.rand(2,1).cuda())
tiv = Variable(torch.rand(2,1).cuda())

outs =model(data,age,tiv)
