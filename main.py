# -*- encoding: utf-8 -*

import tensorflow as tf
import numpy as np
import os
import sys
import re
import pdb

import util
import layer
import network

### Algorithm: fine-tune pre-trained model
## usage: fineTune(sess, trainData, trainLabel, testData, testLabel, **kwargs)
import baseline.fineTune
fineTuneArgs = {
    "batch_size":5,
    "keep_prob":0.4,
    "learning_rate":0.005,
    "learning_rate_decay":0.999
}

### Algorithm: KNeighborsClassifier
## usage: knn(trainData, trainLabel, testData, testLabel, **kwargs)
import baseline.knn
'''
n_neighbors: Number of neighbors to use by default for kneighbors queries.
weights:
    1. 'uniform': uniform weights. All points in each neighborhood are weighted equally.
    2. 'distance' : weight points by the inverse of their distance. in this case, closer
     neighbors of a query point will have a greater influence than neighbors which are further away.
    3. [callable] : a user-defined function which accepts an array of distances, and returns
     an array of the same shape containing the weights.
p: distance L_p. L_2 is Euclidean distance.
PCA: whether use PCA.
n_components: if use PCA, the number of main components.
'''

knnArgs = {
    'n_neighbors': 10,
    'weights': 'distance',
    'p': 2.5,
    'PCA': True,
    'n_components': 100
}

import baseline.bayes
'''
PCA: whether use PCA.
n_components: if use PCA, the number of main components.
'''
bayesArgs = {
    'PCA': True,
    'n_components': 100
}

import baseline.linearRegression

import baseline.svm
svmArgs = {
}

if __name__=="__main__":

    # Initialization
    sess = tf.Session()
    inputData, inputLabel = util.uploadData(sess)
    trainData, trainLabel, testData, testLabel = util.divideData(inputData, inputLabel)

    pdb.set_trace()
    print(trainData.shape, trainLabel.shape, testData.shape, testLabel.shape)
    # trainData = util.normalization(trainData)
    # testData = util.normalization(testData)

    data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
    model = layer.AlexNet(data_, 1, 50, ['fc8'])
    model.load_initial_weights(sess)
    trainData = util.extractFeature(sess, model, trainData)
    testData = util.extractFeature(sess, model, testData)

    # Training & Testing

    # fineTuneAcc = baseline.fineTune.fineTune(sess, trainData, trainLabel, testData, testLabel, **fineTuneArgs)
    # knnAcc = baseline.knn.knn(trainData, trainLabel, testData, testLabel, **knnArgs)
    bayesAcc = baseline.bayes.bayes(trainData, trainLabel, testData, testLabel, **bayesArgs)
    # linearRegAcc = baseline.linearRegression.linearReg(trainData, trainLabel, testData, testLabel)
    # baseline.svm.svm(trainData, trainLabel, testData, testLabel, **svmArgs)
    # pdb.set_trace()