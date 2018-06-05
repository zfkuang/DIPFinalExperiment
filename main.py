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
import baseline.logisticRegression
import baseline.svm
svmArgs = {
}

import baseline.decisionTree
decisionTreeArgs = {
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'alpha': 0,   # L1正则化系数
    'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.5,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.03,  # 如同学习率
    'seed': 1000,
    'nthread': 1,  # cpu 线程数
    'missing': 1
}

if __name__=="__main__":

    # Initialization
    sess = tf.Session()
    inputData, inputLabel = util.uploadData(sess)
    trainData, trainLabel, testData, testLabel = util.divideData(inputData, inputLabel)

    # pdb.set_trace()
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
    # bayesAcc = baseline.bayes.bayes(trainData, trainLabel, testData, testLabel, **bayesArgs)
    # baseline.svm.svm(trainData, trainLabel, testData, testLabel, **svmArgs)
    decisionTreeAcc = baseline.decisionTree.decisionTree(trainData, trainLabel, testData, testLabel, **decisionTreeArgs)
    # logisticRegAcc = baseline.logisticRegression.logisticReg(trainData, trainLabel, testData, testLabel)
    # linearRegAcc = baseline.linearRegression.linearReg(trainData, trainLabel, testData, testLabel)


    # pdb.set_trace()