import os
import torch
import h5py
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from DataAugment import DataAugment
import numpy as np
import pandas as pd
augment = DataAugment()
zeros=0
ones =0
transform = {'MinMax': True, 'ZScore' : True, 'Resize': 82, 'TenCrop': 64}

s1_data = pd.read_csv("/home/uavws/Documents/avinash/PAC18/processed_data/scanner_1.csv")
s23_data = pd.read_csv("/home/uavws/Documents/avinash/PAC18/processed_data/scanner_2_3.csv")
info = pd.read_csv('/home/uavws/Documents/avinash/PAC18/raw_data/PAC_info_sheet.csv')# PAC_info_sheet.csv

files = []
files.extend(s1_data[s1_data['Training']]['Volume_Path'])
files.extend(s23_data[s23_data['Training']]['Volume_Path'])

class_of_interest = [0,1]

scanner_1_model = torch.load("/home/uavws/Documents/avinash/PAC18/models4/model-m-09052018-184328-densenet3D_scanner_1_loss = tensor(0.6248, device='cuda:0') accuracy = 0.669354838709677.pth.tar")
scanner_23_model = torch.load("/home/uavws/Documents/avinash/PAC18/models4/model-m-09052018-190544-densenet3D_scanner_2_3_loss = tensor(0.6887, device='cuda:0') accuracy = 0.6148648648648649.pth.tar")

cntr =0
total_number_of_files=0

pred, gts =[], []

for f in files:
		# f = f[:-4] + '.hdf5'
		id_ = f.split('/').pop().split('.')[0]
		h5 = h5py.File(f,'r')
		numpy_image = h5['volume'][:]
		age = h5['age'][:]/100.0 # normalizing values
		tiv = h5['tiv'][:]/3000.0 # normalizing values
		label = h5['label'][:]
		scanner = int(info[info['PAC_ID'] == id_]['Scanner'])
		print ('currently testing',f)
		print ('file info: {}'.format(id_) + ' scanner info: {}'.format(scanner))

		# apply transforms
		try:
			minmax = transform['MinMax']
			numpy_image = augment.MinMaxNormalization(numpy_image)
		except: pass

		try:
			resize = transform['Resize']
			numpy_image = augment.Resize(numpy_image, resize)
		except: pass

		try:
			random_resize = transform['RandomResizedCrop']
			numpy_image = augment.RandomCrop(numpy_image, random_resize)
			# print numpy_image.shape
		except: pass

		try:
			tencrop = transform['TenCrop']
			numpy_image = augment.TenCrop(numpy_image, tencrop)
		except: pass

		if len(numpy_image.shape) == 3:
			imageData = Variable(torch.from_numpy(np.expand_dims(np.expand_dims(numpy_image, 0), 0)).cuda(), volatile=True)
		else: imageData = Variable(torch.from_numpy(np.expand_dims(numpy_image, 1)).cuda(), volatile=True)

		imageAge = Variable(torch.FloatTensor(age).cuda(), volatile=True)
		imageTiv = Variable(torch.FloatTensor(tiv).cuda(),volatile=True)

		print(imageData.size())

		# prediction code
		gts.append(int(label[0] -1))
		if scanner == 1:
			outs = scanner_1_model(imageData, imageAge, imageTiv)
		else:
			outs = scanner_23_model(imageData, imageAge, imageTiv)

		_,class_associated_to_each_Crop = torch.max(outs,1)
		del _
		del outs

		class_associated_to_each_Crop = class_associated_to_each_Crop.data.cpu().numpy()
		unique,counts = np.unique(class_associated_to_each_Crop,return_counts=True)

		del class_associated_to_each_Crop

		image_pred = unique[np.argmax(counts)] ### one max done
		del unique
		del counts

		pred.append(image_pred)

pred = np.array(pred)
gts = np.array(gts)
conf_mat = np.array([[0, 0], [0, 0]])
conf_mat[0,0] = np.sum(((pred == 0) * (gts == 0)))
conf_mat[1,1] = np.sum(((pred == 1) * (gts == 1)))
conf_mat[0,1] = np.sum(((pred == 0) * (gts == 1)))
conf_mat[1,0] = np.sum(((pred == 1) * (gts == 0)))

print (conf_mat)