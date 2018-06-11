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

import models.binary_classifier
binaryClassifierArgs = {
    "batch_size":5,
    "keep_prob":0.5,
    "learning_rate":0.005,
    "learning_rate_decay":0.999,
    "model":"mlp",
    "epoch":50,
    "lambda_l2": 0.005
}


import models.multi_classifier
multiClassifierArgs = {
    "batch_size":5,
    "keep_prob":0.5,
    "learning_rate":0.001,
    "learning_rate_decay":0.999,
    "model":"mlp",
    "epoch":100,
    "lambda_l2": 0.005
}

### Algorithm: fine-tune pre-trained model
## usage: fineTune(sess, trainData, trainLabel, testData, testLabel, **kwargs)
import baseline.fineTune
fineTuneArgs = {
    "batch_size":20,
    "keep_prob":0.6,
    "learning_rate":0.001,
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

import models.prototypicalNetwork
prototypicalNetworkArgs = {
}

import models.vanerModel
vanerModelArgs = {
    'n' : 1000,             # base classes count
    'q' : 600,              # to be optimized
    'p' : 4096,             # feature count
    'lambda' : 1.0,         # to be optimized
    'learning_rate' : 1e-2,
    'learning_rate_decay' : 0.999
}


if __name__=="__main__":

    # Initialization

    trainData, trainLabel = util.uploadData(3, sampleNumber=500, dataFolder="training", fileNameRegex=r"(?P<group>\d{3})_(?P<index>\d{4}).jpg", groupInFilename=True)
    testData, testLabel = util.uploadData(0, sampleNumber=2500, dataFolder="testing", fileNameRegex=r"testing_(?P<index>\d*).jpg", groupInFilename=False)

    sess = tf.Session()
    #inputData = util.normalization(inputData)
    #trainData, trainLabel, testData, testLabel = util.divideData(inputData, inputLabel)
    basicData, basicLabel, basicIndex = util.uploadBasicData()
    
    print("Data Input Completed!")
    print("    trainDataset shape:", trainData.shape, trainLabel.shape)
    print("    testDataset shape:", testData.shape, testLabel.shape)
    print("    sourceDataset shape:", basicData.shape, basicLabel.shape)

    # trainData = util.normalization(trainData)
    # testData = util.normalization(testData)


    # Feature extraction
    # data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
    # model = layer.AlexNet(data_, 1, 1000, [])
    # model.load_initial_weights(sess)
    # trainData = util.extractFeature(sess, model, trainData)
    # testData = util.extractFeature(sess, model, testData)

    #np.save('train_4096_fc7.npy', trainData)
    #np.save('test_4096_fc7.npy', testData)
    #np.save('trainlabel_350_fc7.npy', trainLabel)
    #np.save('testlabel_150_fc7.npy', testLabel)

    bayesLogit, bayesAcc = baseline.bayes.bayes(trainData, trainLabel, testData, testLabel, **bayesArgs)
    print(bayesLogit, bayesAcc)
    svmLogit, svmAcc = baseline.svm.svm(trainData, trainLabel, testData, testLabel, **svmArgs)
    print(svmLogit, svmAcc)
    knnLogit, knnAcc = baseline.knn.knn(trainData, trainLabel, testData, testLabel, **knnArgs)
    print(knnLogit, knnAcc)
    fineTuneLogit, fineTuneAcc = baseline.fineTune.fineTune(sess, trainData, trainLabel, testData, testLabel, **fineTuneArgs)
    print(fineTuneLogit, fineTuneAcc)
    logisticRegLogit, logisticRegAcc = baseline.logisticRegression.logisticReg(trainData, trainLabel, testData, testLabel)
    print(logisticRegLogit, logisticRegAcc)

    print("\n\nBaseline result: ")
    print("Bayes: ", bayesAcc)
    print("SVM: ", svmAcc)
    print("KNN: ", knnAcc)
    print("FineTune: ", fineTuneAcc)
    print("LogisticRegression: ", logisticRegAcc)
    a = open("data/novalAns.txt").read().split('\n')[:-1]
    for i in range(len(a)):
        a[i] = int(a[i])
    novelLogit = np.array(a)
    pdb.set_trace()
    bayesLogit = bayesLogit.reshape((bayesLogit.shape[0], 1))
    svmLogit = svmLogit.reshape((svmLogit.shape[0], 1))
    knnLogit = knnLogit.reshape((knnLogit.shape[0], 1))
    fineTuneLogit = fineTuneLogit.reshape((fineTuneLogit.shape[0], 1))
    logisticRegLogit = logisticRegLogit.reshape((logisticRegLogit.shape[0], 1))
    novelLogit = novelLogit.reshape((novelLogit.shape[0], 1))
    logits = np.concatenate((bayesLogit, knnLogit, novelLogit, svmLogit, fineTuneLogit, logisticRegLogit), axis=1)
    finalLogit = np.zeros((svmLogit.shape[0]), dtype=np.int32)
    for i in range(finalLogit.shape[0]):
        finalLogit[i] = np.argmax(np.bincount(logits[i]))
    pdb.set_trace()
    #decisionTreeAcc = baseline.decisionTree.decisionTree(trainData, trainLabel, testData, testLabel, **decisionTreeArgs)
    #linearRegAcc = baseline.linearRegression.linearReg(trainData, trainLabel, testData, testLabel)

    # trainData = trainData.reshape(50, 10, 4096)
    # testData = testData.reshape(50, 50, 4096)
    # trainLabel = trainLabel.reshape(50, 10)
    # testLabel = testLabel.reshape(50, 50)
    # inputData = np.concatenate((trainData, testData), axis=1)
    # inputLabel = np.concatenate((trainLabel, testLabel), axis=1)
    # inputData = inputData.reshape(500, 4096)
    # inputLabel = inputLabel.reshape(500)

    #models.prototypicalNetwork.prototypicalNetwork(sess, basicData, basicLabel, basicIndex, trainData, trainLabel, **prototypicalNetworkArgs)

    # pNetwork = models.prototypicalNetwork.prototypicalNetwork(sess)
    # pNetwork.train(sess, basicData, basicLabel, basicIndex, trainData, trainLabel, testData, testLabel, **prototypicalNetworkArgs)
    # tempTrainData = trainData.reshape((50, 10, 4096))
    # pNetwork.inference(sess, tempTrainData, testData)


    #models.binary_classifier.train_base_classifier(sess, basicData, basicLabel, basicIndex, **binaryClassifierArgs)

    # models.binary_classifier.train_base_classifier(sess, basicData, basicLabel, basicIndex, **binaryClassifierArgs)
    # models.binary_classifier.test_base_classifier(sess, basicData, basicLabel, **binaryClassifierArgs)

    # pos_data = basicData[basicIndex[0]]
    # pos_label = [1] * len(basicIndex[0])
    # neg_data = basicData[basicIndex[1]]
    # neg_label = [0] * len(basicIndex[1])
    # data = np.concatenate((pos_data, neg_data))
    # label = np.concatenate((pos_label, neg_label))

    # W = np.load('data/base_fc8.npy').T
    # W = util.loadBaseClassifier()
    # feature_avg = np.load('data/feature_avg.npy')
    # print(feature_avg.shape)
    # models.vanerModel.trainVanerModel(sess, basicData, basicLabel, basicIndex, feature_avg, W, **vanerModelArgs)
    #models.binary_classifier.train_novel_classifier(sess, trainData, trainLabel, testData, testLabel, **binaryClassifierArgs)
    #models.binary_classifier.test_novel_classifier(sess, testData, testLabel, **binaryClassifierArgs)

    #models.multi_classifier.train_novel_classifier(sess, trainData, trainLabel, testData, testLabel, **multiClassifierArgs)
