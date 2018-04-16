import tensorflow as tf
import numpy as np
from utils.config import get
from utils.patches import ExtractImagePatches3D
from shared_placeholders import *
from utils.args import *

def standardBatchNorm(inputs, trainingPL, momentum=0.9, name=None):
    return tf.layers.batch_normalization(inputs, training=trainingPL, momentum=momentum, name=name, reuse=tf.AUTO_REUSE)

def standardPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.max_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def avgPool(inputs, kernel_size=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=None):
    return tf.nn.avg_pool3d(inputs, ksize=kernel_size, strides=strides, padding=padding, name=name)

def standardConvolution(inputs, filters, kernel_size=(3,3,3), activation=tf.nn.elu, strides=(1,1,1), padding='SAME', name=None):
    kernel_regularizer=None
    kernel_constraint=None
    if GlobalOpts.regStrength is not None:
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=GlobalOpts.regStrength)
    if GlobalOpts.maxNorm is not None:
        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=GlobalOpts.maxNorm, axis=[0,1,2,3])
    return tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                            bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def standardDense(inputs, units, activation=tf.nn.elu, use_bias=True, name=None):
    if use_bias:
        kernel_regularizer=None
        kernel_constraint=None
        if GlobalOpts.regStrength is not None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=GlobalOpts.regStrength)
        if GlobalOpts.maxNorm is not None:
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=GlobalOpts.maxNorm, axis=[0])
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           bias_initializer=tf.zeros_initializer(),
                           kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           name=name, reuse=tf.AUTO_REUSE)
    else:
        return tf.layers.dense(inputs=inputs, units=units, activation=activation,
                           use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.zeros_initializer(), name=name, reuse=tf.AUTO_REUSE)

def standardBlock(inputs,
                  trainingPL,
                  blockNumber,
                  filters,
                  kernelSize=(3,3,3),
                  kernelStrides=(1,1,1),
                  normalize=True,
                  poolStrides=[1,2,2,2,1],
                  poolType='MAX',
                  skipConnection=False):
    with tf.variable_scope('ConvBlock{}'.format(blockNumber)):
        BlockConvolution1 = standardConvolution(inputs,
                                                filters=filters,
                                                name='Block{}Convolution1'.format(blockNumber),
                                                kernel_size=kernelSize,
                                                strides=kernelStrides)
        BlockConvolution2 = standardConvolution(BlockConvolution1,
                                                filters=filters,
                                                name='Block{}Convolution2'.format(blockNumber),
                                                kernel_size=kernelSize,
                                                strides=kernelStrides)
        outputLayer = BlockConvolution2

        if normalize:
            outputLayer = standardBatchNorm(outputLayer, trainingPL, name='Block{}BatchNorm'.format(blockNumber))
        if poolType=='MAX':
            outputLayer = standardPool(outputLayer, strides=poolStrides, name='Block{}MaxPool'.format(blockNumber))
        elif poolType=='AVERAGE':
            outputLayer = avgPool(outputLayer, strides=poolStrides, name='Block{}AvgPool'.format(blockNumber))
        if skipConnection:
            if poolType == 'MAX':
                pooledInput = standardPool(inputs, strides=poolStrides, name='Block{}InputMaxPool'.format(blockNumber))
            elif poolType == 'AVERAGE':
                pooledInput = avgPool(inputs, strides=poolStrides, name='Block{}InputMaxPool'.format(blockNumber))
            filteredInput = standardConvolution(pooledInput,
                                                filters=filters,
                                                name='Block{}InputConvolution'.format(blockNumber),
                                                kernel_size=(1,1,1),
                                                strides=(1,1,1))
            outputLayer = outputLayer + filteredInput
        return outputLayer

def padImageTensor(image, padding, cropDims=[[3, 37], [3, 45], [0, 36]]):
    with tf.variable_scope('Padding'):
        croppedImage = image[:,
                             cropDims[0][0]:cropDims[0][1],
                             cropDims[1][0]:cropDims[1][1],
                             cropDims[2][0]:cropDims[2][1],
                             :]
        paddedImage = tf.pad(croppedImage,
                            paddings=[
                                [0, 0],
                                [padding, padding],
                                [padding, padding],
                                [padding, padding],
                                [0, 0]
                            ],
                            name='ZeroPaddingOp')
    return paddedImage
    
def attentionMap(inputs, randomInit=False):
    with tf.variable_scope('attentionMap'):
        weightShape = inputs.shape.as_list()
        weightShape[0] = 1
        if randomInit:
            attentionWeight = tf.Variable(name='attentionWeight',
                                              initial_value=tf.random_normal(shape=weightShape,
                                                   mean=1.0,
                                                   stddev=0.05),
                                              dtype=tf.float32)
        else:
            attentionWeight = tf.Variable(name='attentionWeight',
                                              initial_value=tf.ones(shape=weightShape),
                                              dtype=tf.float32)
        return tf.multiply(inputs, attentionWeight)
