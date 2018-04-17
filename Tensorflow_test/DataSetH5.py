import tensorflow as tf
import os
import numpy as np
import h5py
from DataAugment import DataAugment

augment = DataAugment()
class DataSetNPY(object):
    def __init__(self,
            filenames,
            imageBaseString,
            phenotypeBaseString,
            labelBaseString,
            imageBatchDims,
            labelBatchDims=(-1,1),
            batchSize=4,
            maxItemsInQueue=50,
            shuffle=True,
            augment='none'
        ):

        self.filenames           = filenames
        self.batchSize           = batchSize
        self.imageBatchDims      = imageBatchDims
        self.labelBatchDims      = labelBatchDims
        self.imageBaseString     = imageBaseString
        self.labelBaseString     = labelBaseString
        self.maxItemsInQueue     = maxItemsInQueue
        self.phenotypeBaseString = phenotypeBaseString
        self.phenotypeBatchOperation = None

        # print filenames, maxItemsInQueue, shuffle
        stringQueue = tf.train.string_input_producer(filenames, shuffle=shuffle, capacity=maxItemsInQueue)
        dequeueOp = stringQueue.dequeue_many(batchSize)
        self.dequeueOp = dequeueOp
        print self.dequeueOp
        self.imageBatchOperation = tf.reshape(
                                    tf.py_func(self._loadImages, [dequeueOp], tf.float32),
                                    self.imageBatchDims)

        print self.imageBatchOperation
        self.labelBatchOperation = tf.reshape(
                                    tf.py_func(self._loadLabels, [dequeueOp], tf.float32),
                                    self.labelBatchDims)

        self.augment = augment
        if self.augment != 'none':
            self.CreateAugmentOperations(augmentation=augment)

    def NextBatch(self, sess):
        if self.augment == 'none':
            if self.phenotypeBatchOperation is not None:
                return sess.run([self.imageBatchOperation, self.labelBatchOperation, self.phenotypeBatchOperation])
            else:
                return sess.run([self.imageBatchOperation, self.labelBatchOperation])
        else:
            if self.phenotypeBatchOperation is not None:
                return sess.run([self.augmentedImageOperation, self.labelBatchOperation, self.phenotypeBatchOperation])
            else:
                return sess.run([self.augmentedImageOperation, self.labelBatchOperation])

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation

    def GetRandomBatchOperations(self):
        randomIndexOperation = tf.random_uniform(shape=(self.batchSize,),
                                                dtype=tf.int32,
                                                minval=0,
                                                maxval=len(self.filenames))
        filenameTensor = tf.constant(self.filenames, dtype=tf.string)
        randomFilenames = tf.gather(filenameTensor, randomIndexOperation)
        randomImageBatch = tf.reshape(
            tf.py_func(self._loadImages, [randomFilenames], tf.float32),
            self.imageBatchDims)
        randomLabelBatch = tf.reshape(
            tf.py_func(self._loadLabels, [randomFilenames], tf.float32),
            self.labelBatchDims)
        return randomImageBatch, randomLabelBatch

    def CreateAugmentOperations(self, augmentation='flip'):
        with tf.variable_scope('DataAugmentation'):
            if augmentation == 'flip':
                augmentedImageOperation = tf.reverse(self.imageBatchOperation,
                                                     axis=[1],
                                                     name='flip')
            elif augmentation == 'translate':
                imageRank = 3
                maxPad = 6
                minPad = 0
                randomPadding = tf.random_uniform(shape=(3, 2),
                                                  minval=minPad,
                                                  maxval=maxPad + 1,
                                                  dtype=tf.int32)
                randomPadding = tf.pad(randomPadding, paddings=[[1, 1], [0, 0]])
                paddedImageOperation = tf.pad(self.imageBatchOperation, randomPadding)
                sliceBegin = randomPadding[:, 1]
                sliceEnd = self.imageBatchDims
                augmentedImageOperation = tf.slice(paddedImageOperation,
                                                sliceBegin,
                                                sliceEnd)

            chooseOperation = tf.cond(
                tf.equal(
                    tf.ones(shape=(), dtype=tf.int32),
                    tf.random_uniform(shape=(), dtype=tf.int32, minval=0, maxval=2)
                ),
                lambda: augmentedImageOperation,
                lambda: self.imageBatchOperation,
                name='ChooseAugmentation'
            )
            self.augmentedImageOperation = tf.reshape(chooseOperation, self.imageBatchDims)
    
    def CreatePhenotypicOperations(self, phenotypicBaseStrings):
        self.phenotypicBaseStrings = phenotypicBaseStrings
        self.phenotypeBatchOperation = tf.reshape(
            tf.py_func(self._loadPhenotypes, [self.dequeueOp], tf.float32),
            (self.batchSize, len(self.phenotypicBaseStrings)))
        
    def _loadPhenotypes(self, x):
        #NOTE: ASSUMES THE FIRST PHENOTYPE IS GENDER, Age,Gender,TIV
        #WHERE MALE IS 1 AND FEMALE IS 2
        #WHERE AGE IS SECOND PHENOTYPE
        #WHERE TIVIS OUR THIRD PHENOTYPE

        phenotypes = np.zeros((self.batchSize, len(self.phenotypicBaseStrings)), dtype=np.float32)
        batchIndex = 0
        for name in x:
            # path = os.path.join(self.phenotypeBaseString, name)
            path = name
            h5 = h5py.File(path, 'r')
            phenotype_data = []
            for pheno_str in self.phenotypicBaseStrings:
                # print "phenotype test: ", pheno_str, h5[pheno_str][:]
                phenotype_data.append(h5[pheno_str][:])
            phenotypes[batchIndex] = phenotype_data
            batchIndex += 1
        return phenotypes
            
    def _loadImages(self, x):
        # TODO: To dynamically control the resize 
        # and randomCrop inputs

        volumes = []
        for name in x:
            # path = os.path.join(self.imageBaseString, name)
            path = name
            h5 = h5py.File(path, 'r')
            vol = h5['volume'][:]
            vol = augment.MinMaxNormalization(vol)
            vol = augment.Resize(vol, 84)
            vol = augment.RandomCrop(vol, 64)
            volumes.append(vol.astype(np.float32))
        volumes = np.array(volumes)
        return volumes

    def _loadLabels(self, x):

        labels = []
        for name in x:
            # path = os.path.join(self.labelBaseString, name)
            path = name
            h5 = h5py.File(path, 'r')
            labels.append(h5['label'][:].astype(np.float32))
        labels = np.array(labels)
        return labels

if __name__ == '__main__':
    dataset = DataSetNPY(filenames=next(os.walk('../processed_data/hdf5_file/'))[2],
                         imageBatchDims=(-1, 64, 64, 64, 1),
                         imageBaseString='../processed_data/hdf5_file/', 
                         labelBaseString='../processed_data/hdf5_file/',
                         phenotypeBaseString='../processed_data/hdf5_file/',  
                         batchSize=2)

    imageOp, labelOp = dataset.GetBatchOperations()
    dequeueOp = dataset.dequeueOp
    # print dequeueOp
    dataset.CreatePhenotypicOperations(phenotypicBaseStrings=[
        'gender',
        'age',
        'tiv'
    ])

    phenotypeOp = dataset.phenotypeBatchOperation
    config  = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(100):
            print "+---------------------------------------------------"
            subjects, images, labels, phenotypes = sess.run([dequeueOp, imageOp, labelOp, phenotypeOp])
            print(subjects)
            print(images.shape)
            print(labels)
            print(phenotypes)
        coord.request_stop()
        coord.join(threads)