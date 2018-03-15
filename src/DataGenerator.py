import torch
from torch.utils.data import Dataset

from PIL import Image
import os
import os.path
import numpy as np 

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.hdf5'
]


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
def getDataFromPath(path):
	h5 = h5py.File(path,'r')
	img = h5['volume'][:]
	print img.shape
	# add data augmentation....
	# img, lbl, weight = getPatchSize(img, lbl,weight)
	return img

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

	def __init__(self, imgs, classes, transform=None, loader=hdf5_loader):

		self.listImagePaths = imgs
		self.listImageLabels = classes
		self.transform = transform
		self.loader = loader

		# sanity check...
		# self.listImagePaths = self.listImagePaths[:5]
		# self.listImageLabels = self.listImageLabels[:5]

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is class_index of the target class.
		"""
		imagePath = self.listImagePaths[index]
		
		imageData = self.loader(imagePath)
		imageLabel= torch.FloatTensor(self.listImageLabels[index])
		
		# print imagePath, np.array(imageData).shape
		# print 
		if self.transform != None: imageData = self.transform(imageData)
		
		return imageData, imageLabel, imagePath

	def __len__(self):

		return len(self.listImagePaths)
