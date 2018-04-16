import tensorflow as tf
import numpy as np
import pandas as pd
from utils.args import *
from DataSetH5 import DataSetNPY
from buildCustomCNN import customCNN
from utils.saveModel import *
from utils.config import get
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
                                      batchSize=GlobalOpts.batchSize,
                                      augment=GlobalOpts.augment)

        with tf.variable_scope('ValidationInputs'):
            valdDataSet  = DataSetNPY(filenames=GlobalOpts.valdFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberValdItems,
                                    shuffle=False)

        with tf.variable_scope('TestInputs'):
            testDataSet  = DataSetNPY(filenames=GlobalOpts.testFiles,
                                    imageBaseString=GlobalOpts.imageBaseString,
                                    imageBatchDims=GlobalOpts.imageBatchDims,
                                    labelBaseString=GlobalOpts.labelBaseString,
                                    batchSize=1,
                                    maxItemsInQueue=GlobalOpts.numberTestItems,
                                    shuffle=False)

    return trainDataSet, valdDataSet, testDataSet

def DefineDataOpts(summaryName='test_comp'):
    GlobalOpts.imageBatchDims = (-1, 64, 64, 64, 1)
 
    GlobalOpts.trainFiles = np.load(get('DATA.TRAIN_LIST')).tolist()
    GlobalOpts.valdFiles = np.load(get('DATA.VALD_LIST')).tolist()
    GlobalOpts.testFiles = np.load(get('DATA.TEST_LIST')).tolist()

    GlobalOpts.imageBaseString       = get('DATA.STRUCTURAL.EXTRA_SMALL_PATH')
    GlobalOpts.phenotypeBaseString   ='../processed_data/hdf5_file/'    
    GlobalOpts.labelBaseString       = get('DATA.LABELS')

    GlobalOpts.numberTrainItems = len(GlobalOpts.trainFiles)
    GlobalOpts.numberTestItems  = len(GlobalOpts.testFiles)
    GlobalOpts.numberValdItems  = len(GlobalOpts.valdFiles)
    GlobalOpts.poolType = 'MAX'

    GlobalOpts.name = '{} Scale: {}  Data: {}  Batch: {}  Rate: {}'.format(GlobalOpts.type, GlobalOpts.scale, data, GlobalOpts.batchSize, GlobalOpts.learningRate)
   
    if GlobalOpts.padding is not None:
        GlobalOpts.name = '{}Padding{}'.format(GlobalOpts.name, GlobalOpts.padding)
    if GlobalOpts.regStrength is not None:
        GlobalOpts.name = '{}L2Reg{}'.format(GlobalOpts.name, GlobalOpts.regStrength)
    if GlobalOpts.maxNorm is not None:
        GlobalOpts.name = '{}MaxNorm{}'.format(GlobalOpts.name, GlobalOpts.maxNorm)
    if GlobalOpts.dropout is not None:
        GlobalOpts.name = '{}Dropout{}'.format(GlobalOpts.name, GlobalOpts.dropout)

    GlobalOpts.summaryDir = '../summaries/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.checkpointDir = '../checkpoints/{}/{}/'.format(summaryName,
                                                     GlobalOpts.name)
    GlobalOpts.augment = 'none'

def GetOps(labelsPL, outputLayer, learningRate=0.0001):
    with tf.variable_scope('LossOperations'):
        lossOp = tf.losses.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MSEOp, MSEUpdateOp = tf.metrics.mean_squared_error(labels=labelsPL, predictions=outputLayer)
        MAEOp, MAEUpdateOp = tf.metrics.mean_absolute_error(labels=labelsPL, predictions=outputLayer)
        updateOp, gradients = GetTrainingOperation(lossOp, learningRate)
        
    printOps = PrintOps(ops=[MSEOp, MAEOp],
        updateOps=[MSEUpdateOp, MAEUpdateOp],
        names=['loss', 'MAE'],
        gradients=gradients)

    return lossOp, printOps, updateOp

def GetArgs():
    additionalArgs = [
        {
        'flag': '--scale',
        'help': 'The scale at which to slice dimensions. For example, a scale of 2 means that each dimension will be devided into 2 distinct regions, for a total of 8 contiguous chunks.',
        'action': 'store',
        'type': int,
        'dest': 'scale',
        'required': True
        },
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
        'flag': '--data',
        'help': 'One of: PNC, PNC_GENDER, ABIDE1, ABIDE2, ABIDE2_AGE',
        'action': 'store',
        'type': str,
        'dest': 'data',
        'required': True
        },
        {
        'flag': '--sliceIndex',
        'help': 'Set this to an integer to select a single brain region as opposed to concatenating all regions along the depth channel.',
        'action': 'store',
        'type': int,
        'dest': 'sliceIndex',
        'required': False,
        'const': None
        },
        {
        'flag': '--align',
        'help': 'Set to true to align channels.',
        'action': 'store',
        'type': int,
        'dest': 'align',
        'required': False,
        'const': None
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
        'flag': '--padding',
        'help': 'Set this to an integer to crop the image to the brain and then apply `padding` amount of padding.',
        'action': 'store',
        'type': int,
        'dest': 'padding',
        'required': False,
        'const': None
        },
        {
        'flag': '--batchSize',
        'help': 'Batch size to train with. Default is 4.',
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
    DefineDataOpts(data=GlobalOpts.data, summaryName=GlobalOpts.summaryName)
    modelTrainer = ModelTrainer()
    trainDataSet, valdDataSet, testDataSet = GetDataSetInputs()
    imagesPL, labelsPL = StructuralPlaceholders(GlobalOpts.imageBatchDims)
    trainingPL = TrainingPlaceholder()

    if GlobalOpts.type == 'traditional':
        convLayers = [8, 16, 32, 64]
    elif GlobalOpts.type == 'reverse':
        convLayers = [64, 32, 16, 8]
    if GlobalOpts.data == 'PNC' or 'AGE' in GlobalOpts.data:
        fullyConnectedLayers = [256, 1]
    else:
        fullyConnectedLayers = [256, 2]
    if GlobalOpts.pheno:
        phenotypicBaseStrings=[
            '/data/psturm/ABIDE/ABIDE2/gender/',
            '/data/psturm/ABIDE/ABIDE2/IQData/FIQ/',
            '/data/psturm/ABIDE/ABIDE2/IQData/VIQ/',
            '/data/psturm/ABIDE/ABIDE2/IQData/PIQ/'
        ]
        if GlobalOpts.data != 'ABIDE2_AGE':
            phenotypicBaseStrings.append('/data/psturm/ABIDE/ABIDE2/ages/')
        phenotypicsPL = tf.placeholder(dtype=tf.float32, shape=(None, len(phenotypicBaseStrings) + 1), name='phenotypicsPL')
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
                            sliceIndex=GlobalOpts.sliceIndex,
                            align=GlobalOpts.align,
                            padding=GlobalOpts.padding,
                            phenotypicsPL=phenotypicsPL)
    lossOp, printOps, updateOp = GetOps(labelsPL, outputLayer, learningRate=GlobalOpts.learningRate)
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