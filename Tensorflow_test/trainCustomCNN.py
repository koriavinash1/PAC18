import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from DataSetH5 import DataSetNPY
from buildCustomCNN import customCNN
from utils.saveModel import *
from trainCommon import *
from shared_placeholders import *

def GetTrainingOperation(lossOp, learningRate):
    with tf.variable_scope('optimizer'):
        if GlobalOpts.regStrength is not None:
            regularizerLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            lossOp = tf.add_n([lossOp] + regularizerLosses, name="RegularizedLoss")
        updateOp, gradients = AdamOptimizer(lossOp, learningRate)
    return updateOp, gradients

def GetDataSetInputs():
    with tf.variable_scope('Inputs'):
        with tf.variable_scope('TrainingInputs'):
            trainDataSet = DataSetNPY(filenames=GlobalOpts.trainFiles,
                                      imageBaseString=GlobalOpts.imageBaseString,
                                      imageBatchDims=GlobalOpts.imageBatchDims,
                                      labelBaseString=GlobalOpts.labelBaseString,
                                      phenotypeBaseString='../processed_data/hdf5_file/',
                                      batchSize=GlobalOpts.batchSize,
                                      augment=GlobalOpts.augment)

        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    phenotypeBaseString='../processed_data/hdf5_file/',
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberValdItems,
                                    shuffle=False)

        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    phenotypeBaseString='../processed_data/hdf5_file/',
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberTestItems,
                                    shuffle=False)

    return trainDataSet, valdDataSet, testDataSet

def DefineDataOpts(summaryName='test_comp'):
    GlobalOpts.imageBatchDims = (-1, 64, 64, 64, 1)
 
    d = pd.read_csv('../processed_data/train_test_split.csv')

    GlobalOpts.trainFiles = d[d['Training']]['Volume Path'].as_matrix().tolist()
    GlobalOpts.valdFiles = d[d['Validation']]['Volume Path'].as_matrix().tolist()
    GlobalOpts.testFiles = d[d['Testing']]['Volume Path'].as_matrix().tolist()

    GlobalOpts.imageBaseString       = '../processed_data/hdf5_file/'
    GlobalOpts.phenotypeBaseString   = '../processed_data/hdf5_file/'    
    GlobalOpts.labelBaseString       = '../processed_data/hdf5_file/'

    GlobalOpts.numberTrainItems = len(GlobalOpts.trainFiles)
    GlobalOpts.numberTestItems  = len(GlobalOpts.testFiles)
    GlobalOpts.numberValdItems  = len(GlobalOpts.valdFiles)
    GlobalOpts.poolType = 'MAX'

    GlobalOpts.name = '{} Scale: {}  Batch: {}  Rate: {}  '.format(GlobalOpts.type, GlobalOpts.scale, GlobalOpts.batchSize, GlobalOpts.learningRate)
   
    if GlobalOpts.padding is not None:
        GlobalOpts.name = '{}Padding:  {}'.format(GlobalOpts.name, GlobalOpts.padding)
    if GlobalOpts.maxNorm is not None:
        GlobalOpts.name = '{}MaxNorm:  {}'.format(GlobalOpts.name, GlobalOpts.maxNorm)
    if GlobalOpts.dropout is not None:
        GlobalOpts.name = '{}Dropout:  {}'.format(GlobalOpts.name, GlobalOpts.dropout)

    GlobalOpts.summaryDir = '../summaries/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '../checkpoints/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.augment = 'flip'

def GetOps(labelsPL, outputLayer, learningRate=0.0001):
    with tf.variable_scope('LossOperations'):
        print labelsPL, outputLayer
        # correct_predictions = tf.equal(tf.argmax(labelsPL,1), tf.argmax(outputLayer,1))
        # accOp = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MSEOp, MSEUpdateOp = tf.metrics.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MAEOp, MAEUpdateOp = tf.metrics.mean_absolute_error(labels=labelsPL, predictions=outputLayer)
        updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        accOp = 0

    printOps = PrintOps(ops=[MSEOp, MAEOp],
        updateOps=[MSEUpdateOp, MAEUpdateOp],
        names=['loss', 'MAE'],
        gradients=gradients)

    return accOp, lossOp, printOps, updateOp

def GetArgs():
    additionalArgs = [
        # {
        # 'flag': '--scale',
        # 'help': 'The scale at which to slice dimensions. For example, a scale of 2 means that each dimension will be devided into 2 distinct regions, for a total of 8 contiguous chunks.',
        # 'action': 'store',
        # 'type': int,
        # 'dest': 'scale',
        # 'required': True
        # },
        {
        'flag': '--type',
        'help': 'One of: traditional, reverse',
        'action': 'store',
        'type': str,
        'dest': 'type',
        'required': True
        },
        {
        'flag': '--summaryName',
        'help': 'The file name to put the results of this run into.',
        'action': 'store',
        'type': str,
        'dest': 'summaryName',
        'required': True
        },
        {
        'flag': '--numberTrials',
        'help': 'Number of repeated models to run.',
        'action': 'store',
        'type': int,
        'dest': 'numberTrials',
        'required': False,
        'const': None
        },
        {
        'flag': '--batchSize',
        'help': 'Batch size to train with. Default is 2.',
        'action': 'store',
        'type': int,
        'dest': 'batchSize',
        'required': False,
        'const': None
        },
        {
        'flag': '--validationDir',
        'help': 'Checkpoint directory to restore the model from.',
        'action': 'store',
        'type': str,
        'dest': 'validationDir',
        'required': False,
        'const': None
        },
        {
        'flag': '--learningRate',
        'help': 'Global optimization learning rate. Default is 0.0001.',
        'action': 'store',
        'type': float,
        'dest': 'learningRate',
        'required': False,
        'const': None
        },
        {
        'flag': '--maxNorm',
        'help': 'Specify an integer to constrain kernels with a maximum norm.',
        'action': 'store',
        'type': int,
        'dest': 'maxNorm',
        'required': False,
        'const': None
        },
        {
        'flag': '--dropout',
        'help': 'The probability of keeping a neuron alive during training. Defaults to 0.6.',
        'action': 'store',
        'type': float,
        'dest': 'dropout',
        'required': False,
        'const': None
        }
        ]
    ParseArgs('Run 3D CNN over structural MRI volumes', additionalArgs=additionalArgs)

    GlobalOpts.scale = 1
    GlobalOpts.padding = None
    GlobalOpts.regStrength = None
    GlobalOpts.pheno = True
    if GlobalOpts.summaryName is None:
        GlobalOpts.summaryName = "test"
    if GlobalOpts.type is None:
        GlobalOpts.type = 'traditional'
    if GlobalOpts.numberTrials is None:
        GlobalOpts.numberTrials = 5
    if GlobalOpts.batchSize is None:
        GlobalOpts.batchSize = 2
    if GlobalOpts.learningRate is None:
        GlobalOpts.learningRate = 0.0001
    if GlobalOpts.dropout is None:
        GlobalOpts.dropout = 0.6

def compareCustomCNN(validate=False):
    GetArgs()
    DefineDataOpts(summaryName=GlobalOpts.summaryName)
    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]

    fullyConnectedLayers = [256, 1]
    if GlobalOpts.pheno:
        phenotypicBaseStrings=[
            'gender',
            'age',
            'tiv'
        ]
        phenotypicsPL = tf.placeholder(dtype=tf.float32, shape=(None, len(phenotypicBaseStrings)), name='phenotypicsPL')
        trainDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
        valdDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
        testDataSet.CreatePhenotypicOperations(phenotypicBaseStrings)
    else:
        phenotypicsPL = None

    outputLayer = customCNN(imagesPL,
                            trainingPL,
                            GlobalOpts.scale,
                            convLayers,
                            fullyConnectedLayers,
                            keepProbability=GlobalOpts.dropout,
                            poolType=GlobalOpts.poolType,
                            # sliceIndex=GlobalOpts.sliceIndex,
                            # align=GlobalOpts.align,
                            # padding=GlobalOpts.padding,
                            phenotypicsPL=phenotypicsPL)

    accuracyOp, lossOp, printOps, updateOp = GetOps(labelsPL, outputLayer, learningRate=GlobalOpts.learningRate)
    modelTrainer.DefineNewParams(GlobalOpts.summaryDir,
                                GlobalOpts.checkpointDir,
                                imagesPL,
                                trainingPL,
                                labelsPL,
                                trainDataSet,
                                valdDataSet,
                                testDataSet,
                                GlobalOpts.numSteps,
                                phenotypicsPL=phenotypicsPL)
    config  = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GlobalOpts.gpuMemory
    with tf.Session(config=config) as sess:
        if validate:
            modelTrainer.ValidateModel(sess,
                                  updateOp,
                                  printOps,
                                  name=GlobalOpts.name,
                                  numIters=GlobalOpts.numberTrials)
        else:
            modelTrainer.RepeatTrials(sess,
                                  updateOp,
                                  printOps,
                                  name=GlobalOpts.name,
                                  numIters=GlobalOpts.numberTrials)


if __name__ == '__main__':
    compareCustomCNN()
    # compareCustomCNN(validate=True)