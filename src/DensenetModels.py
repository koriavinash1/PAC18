import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet3D(nn.Module):
	def __init__(self, num_classes, dropRate=0.8):
		super(DenseNet3D, self).__init__()
		self.droprate = dropRate

		self.conv1 = nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=0, bias=False)
		torch.nn.init.xavier_uniform(self.conv1.weight)
		self.relu1 = nn.ReLU(inplace=True)
		
		self.conv2 = nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0, bias=False)
		torch.nn.init.xavier_uniform(self.conv2.weight)
		self.relu2 = nn.ReLU(inplace=True)

		self.fc1 = nn.Linear(8788 , 1024)
		torch.nn.init.xavier_uniform(self.fc1.weight)
		self.relu3 = nn.ReLU(inplace=True)

		self.fc2 = nn.Linear(1024, 256)
		torch.nn.init.xavier_uniform(self.fc2.weight)
		self.relu4 = nn.ReLU(inplace=True)

		self.fc3 = nn.Linear(256, num_classes)
		torch.nn.init.xavier_uniform(self.fc3.weight)
		self.relu5 = nn.ReLU(inplace=True)

		self.softmax = nn.Softmax()


	def forward(self, x):
		out = self.relu1(self.conv1(x))
		out = F.max_pool3d(out, 2)
		out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

		out = self.relu2(self.conv2(out))
		out = F.max_pool3d(out, 2)
		out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

		out = out.view(out.size(0), -1)
		out = self.relu3(self.fc1(out))
		out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

		out = self.relu4(self.fc2(out))
		out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

		out = self.fc3(out)	
		return self.softmax(out)