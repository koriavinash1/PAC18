import torch
from torch.utils.data import Dataset
from DataAugment import DataAugment
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import h5py
import numpy as np 

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.hdf5'
]

ALLOWED_TRANSFORMS = [
	'Resize', 'RandomResizedCrop', 'HorizontalFlip', 'RandomFlip', 'TenCrop',\
	 'FiveCrop', 'MinMax', 'ZScore', 'HistogramEquilization', 'Sequence'
]

augment = DataAugment()
#----------------------------------------------------------------------------------------------------------------------------

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}
	return classes, class_to_idx


def make_dataset(dir, class_to_idx):
	images = []
	dir = os.path.expanduser(dir)
	for target in sorted(os.listdir(dir)):
		d = os.path.join(dir, target)
		if not os.path.isdir(d):
			continue

		for root, _, fnames in sorted(os.walk(d)):
			for fname in sorted(fnames):
				if is_image_file(fname):
					path = os.path.join(root, fname)
					item = (path, class_to_idx[target])
					images.append(item)

	return images


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def numpy_loader(path):
	x=np.load(path)
	# print x.shape
	x=np.swapaxes(x,0,2)
	x= np.swapaxes(x,0,1)
	# print 'now'
	# print x.shape
	return x

# TODO:
def hdf5_loader(path):
	h5 = h5py.File(path,'r')
	img = h5['volume'][:]
	age = h5['age'][:]/100.0 # normalizing values
	tiv = h5['tiv'][:]/3000.0 # normalizing values
	label = h5['label'][:]
	return img, age, tiv, label

def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)

#-------------------------------------------------------------------------------- 
def one_hot(val, nclasses):
	a = np.zeros(nclasses)
	a[int(val) - 1] = 1
	return a


class DatasetGenerator(Dataset):
	"""A generic data loader where the images are arranged in this way: ::

		../Processed_data/hdf5_files/abc.hdf5

	Args:
		imgs (list of strings): list of all directory paths.
		classes (list): list of corresponding classes
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		loader (callable, optional): A function to load an image given its path.

	 Attributes:
		classes (list): List of the class names.
		class_to_idx (dict): Dict with items (class_name, class_index).
		imgs (list): List of (image path, class_index) tuples
	"""

	def __init__(self, imgs, transform=None, loader=hdf5_loader):

		self.listImagePaths = imgs
		self.transform = transform
		self.loader = loader

		print (self.transform)
		for key in self.transform:
			if key not in ALLOWED_TRANSFORMS:
				raise ValueError('Unknown Transformed Included, \
					Allowed Transformations are: {}'.format(ALLOWED_TRANSFORMS))


		# sanity check...
		print (len(self.listImagePaths))
		# self.listImagePaths = self.listImagePaths[:50]
		# # self.listImageLabels = self.listImageLabels[:50]

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is class_index of the target class.
		"""
		imagePath = self.listImagePaths[index]
		numpy_image, age, tiv, label  = self.loader(imagePath)

		# Augmentation parameteres.....
		try:
			minmax = self.transform['MinMax']
			numpy_image = augment.MinMaxNormalization(numpy_image)
		except: pass

		try:
			minmax = self.transform['ZScore']
			numpy_image = augment.ZScore(numpy_image)
		except: pass

		try:
			resize = self.transform['Resize']
			numpy_image = augment.Resize(numpy_image, resize)
		except: pass

		try:
			random_resize = self.transform['RandomResizedCrop']
			numpy_image = augment.RandomCrop(numpy_image, random_resize)
			# print numpy_image.shape
		except: pass
		
		try:
			tencrop = self.transform['TenCrop']
			numpy_image = augment.TenCrop(numpy_image, tencrop)
		except: pass

		if len(numpy_image.shape) == 3:
			imageData = torch.from_numpy(np.expand_dims(numpy_image, 0))
		else: imageData = torch.from_numpy(np.expand_dims(numpy_image, 1))

		imageLabel = torch.FloatTensor(one_hot(label, nclasses=2))
		imageAge = torch.FloatTensor(age)
		imageTiv = torch.FloatTensor(tiv)

		# print imageData.size()
		# print imagePath, imageData.size(), imageLabel, imageTiv, imageAge, imageGender
		
		return imageData, imageLabel, imageAge, imageTiv, imagePath

	def __len__(self):

		return len(self.listImagePaths)





"""
if __name__ == '__main__':
	transformList = {}
	transformList['Resize'] = imgtransResize

	# any one of these TenCrop and FiveCrop should be True......
	transformList['TenCrop'] = True
	transformList['TenCropSize'] = imgtransCrop

	transformList['Sequence'] = True
	
	datasetTest = DatasetGenerator(pathImageDirectory=pathTestData, transform=transformList)
	dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
	for batchID, (input, target, _) in tqdm(enumerate (dataLoader)):
		# print 		
		target = target.cpu()
		input = input.cpu()
		print input.shape, target.shape
"""
