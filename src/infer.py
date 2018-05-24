import os
import torch
import h5py
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from DataAugment import DataAugment
import numpy as np
import nibabel as nib
import pandas as pd

augment = DataAugment()
transform = {'MinMax': True, 'ZScore' : True, 'Resize': 82, 'TenCrop': 64}

# transform = {'MinMax': True, 'Resize': 82, 'TenCrop': 64}# allowed transformations

root_data_path = '/home/uavws/Documents/avinash/PAC18/raw_data/data_testset_shuffled'
info = pd.read_csv('/home/uavws/Documents/avinash/PAC18/raw_data/PAC2018_Covariates_Testset.csv')# test PAC_info_sheet.csv

files = info['PAC_ID'].as_matrix()

scanner_1_model = torch.load("/home/uavws/Documents/avinash/PAC18/models4/model-m-09052018-184328-densenet3D_scanner_1_loss = tensor(0.6248, device='cuda:0') accuracy = 0.669354838709677.pth.tar")
scanner_23_model = torch.load("/home/uavws/Documents/avinash/PAC18/models4/model-m-09052018-190544-densenet3D_scanner_2_3_loss = tensor(0.6887, device='cuda:0') accuracy = 0.6148648648648649.pth.tar")


predictions  = []
print (transform)

for id_ in files:
	path 				= os.path.join(root_data_path, (id_ + '.nii'))
	numpy_image 		= nib.load(path).get_data()
	idx 				= info[info['PAC_ID'] == id_]
	age 				= np.array([idx['Age'].values/100.0]) # normalizing values
	tiv 				= np.array([idx['TIV'].values/3000.0]) # normalizing values
	scanner 			= int(idx['Scanner'].values)

	print ('file info: {}'.format(id_) + ' scanner info: {}'.format(scanner))

	# apply transforms
	try:
		minmax      = transform['MinMax']
		numpy_image = augment.MinMaxNormalization(numpy_image)
	except: pass

	try:
		resize      = transform['Resize']
		numpy_image = augment.Resize(numpy_image, resize)
	except: pass

	try:
		random_resize = transform['RandomResizedCrop']
		numpy_image   = augment.RandomCrop(numpy_image, random_resize)
	except: pass

	try:
		tencrop     = transform['TenCrop']
		numpy_image = augment.TenCrop(numpy_image, tencrop)
	except: pass

	if len(numpy_image.shape) == 3:
		imageData = Variable(torch.FloatTensor(np.expand_dims(np.expand_dims(numpy_image, 0), 0)).cuda(), volatile=True)
	else: imageData = Variable(torch.FloatTensor(np.expand_dims(numpy_image, 1)).cuda(), volatile=True)

	imageAge = Variable(torch.FloatTensor(age).cuda(), volatile=True)
	imageTiv = Variable(torch.FloatTensor(tiv).cuda(),volatile=True)


	# prediction code
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

	predictions.append(image_pred + 1)

sub               = pd.DataFrame()
sub['PAC_ID']     = files
sub['Prediction'] = predictions
sub.to_csv('submission_pac2018.csv')
