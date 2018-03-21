from PIL import Image
import numpy as np
import torch
import SimpleITK as sitk
import torch.nn as nn
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DataAugment(object):
	#----  Consists all data_augmentation fucnitons

	def Resize(self, volume, out_size): 
		# out_size: int, function adjust's aspect ratio
		l, w, h = volume.shape
		l, w, h = float(l), float(w), float(h)

		if h < l and h < w: 
			Ah = out_size
			Aw = (Ah/h) * w
			Al = (Ah/h) * l

		elif l < w:  
			Al = out_size
			Aw = (Al/l) * w
			Ah = (Al/l)  * h

		else : 
			Aw = out_size
			Al = (Aw/w) * l
			Ah = (Aw/w) * h
		

		# Al, Aw = out_size, out_size
		# Ah = Al * h/l

		Al, Aw, Ah = int(Al), int(Aw), int(Ah)
		# print l,w,h, Al, Aw, Ah
		resized_volume = self.resize_sitk_3D(volume, 
					outputSize= (Al, Aw, Ah))

		return resized_volume

	def resize_sitk_3D(self, image_array, outputSize=None, interpolator=sitk.sitkLinear):
		"""
		Resample 3D images Image:
		For Labels use nearest neighbour
		For image use 
		sitkNearestNeighbor = 1,
		sitkLinear = 2,
		sitkBSpline = 3,
		sitkGaussian = 4,
		sitkLabelGaussian = 5, 
		"""
		image = sitk.GetImageFromArray(image_array) 
		inputSize = image.GetSize()
		inputSpacing = image.GetSpacing()
		outputSpacing = [1.0, 1.0, 1.0]
		if outputSize:
			outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
			outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
			outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
		else:
			# If No outputSize is specified then resample to 1mm spacing
			outputSize = [0.0, 0.0, 0.0]
			outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
			outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
			outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
		resampler = sitk.ResampleImageFilter()
		resampler.SetSize(outputSize)
		resampler.SetOutputSpacing(outputSpacing)
		resampler.SetOutputOrigin(image.GetOrigin())
		resampler.SetOutputDirection(image.GetDirection())
		resampler.SetInterpolator(interpolator)
		resampler.SetDefaultPixelValue(0)
		image = resampler.Execute(image)
		resampled_arr = sitk.GetArrayFromImage(image)
		return resampled_arr

	def Crop(self, volume, x,y,z, size):
		return volume[x:x+size, y:y+size, z:z+size]

	def RandomCrop(self, volume, size):
		"""
		Args: 
			volume: numpy object
			size: int for cubical volume output
			TODO: cuboidal output
		Return:
			volume of specified size: numpy object
		"""
		l, w, h = volume.shape
		x = np.random.randint(0, l-size)
		y = np.random.randint(0, w-size)
		z = np.random.randint(0, h-size)
		img = self.Crop(volume, x, y, z, size)

		return img

	def TenCrop(self, volume, size):
		"""
		Args: 
			volume: numpy object
			size: int for cubical volume output
			TODO: cuboidal output
		Return:
			Ten volumes of specified size: numpy object
		"""
		x, y, z  = volume.shape
		volumes = []
		# front five...
		volumes.append(self.Crop(volume, 0, y-size, z-size, size))
		volumes.append(self.Crop(volume, 0, y-size, 0, size))
		volumes.append(self.Crop(volume, x-size, y-size, 0, size))
		volumes.append(self.Crop(volume, x-size, y-size, z-size, size))
		volumes.append(self.Crop(volume, (x-size)//2, y-size, (z-size)//2, size))

		# back five...
		volumes.append(self.Crop(volume, 0, 0, 0, size))
		volumes.append(self.Crop(volume, 0, 0, z-size, size))
		volumes.append(self.Crop(volume, x-size, 0, z-size, size))
		volumes.append(self.Crop(volume, x-size, 0, 0, size))
		volumes.append(self.Crop(volume, (x-size)//2, 0, (z-size)//2, size))

		return np.array(volumes)

	def MinMaxNormalization(self, volume):
		"""
		Args:
			volume: numpy object
		return:
			volume of same size as input: numpy object

		Converts the range of intensity values between 0-1.0
		"""
		volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
		return volume

	def ZScore(self, volume):

		"""
		Args:
			volume: numpy object
		return:
			volume of same size as input: numpy object

		Converts the statistics of image intensity values : mean=0, std=1.0
		"""
		volume = (volume - np.mean(volume)) / np.std(volume)
		return volume

	def HistogramEquilization(self, volume):
		return volume