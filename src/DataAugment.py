from PIL import Image
import numpy as np
import torch
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F

class DataAugment(object):
	#----  Consists all data_augmentation fucnitons

	def Resize(self, volume, out_size): 
		# out_size: int, function adjust's aspect ratio
		l, w, h = volume.shape
		if h < l and h < w: 
			Ah= out_size
			if I < w: Al = Ah/h * l
			else : Aw = Ah/h * Al/l * w

		elif l < w:  
			Al = out_size
			if h < w: Ah = Al/l * h
			else : Aw = Ah/h * Al/l * w

		else : 
			Aw = out_size
			if h < I: Ah = Aw/w * l
			else : Al = Aw/w * Ah/h * l

		Al, Aw, Ah = int(Ai), int(Aw), int(Ah)
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

	def RandomCrop(self, volume, size):
		"""
		Args: 
			volume: numpy object
			size: int for cubical volume output
			TODO: cuboidal output
		Return:
			volume: numpy object
		"""
		l, w, h = volume.shape
		x = np.random.randint(0, l-size)
		y = np.random.randint(0, w-size)
		z = np.random.randint(0, h-size)

		return volume[x:x+size, y:y+size, z:z+size]
