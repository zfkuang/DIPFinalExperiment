# -*- encoding: utf-8 -*

from sklearn import linear_model
import numpy as np

import util

'''
Method:
1、预处理数据：对数据进行进行归一化处理，使数据均值为0，方差为1。
3、分割训练集，要有验证集。
'''

def linearReg(trainData, trainLabel, testData, testLabel, **kwargs):
    print(kwargs)
    # neigh = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'], weights=kwargs['weights'], p=kwargs['p'])
    clf = linear_model.LinearRegression()
    trainData = util.normalization(trainData)
    testData = util.normalization(testData)
    acc_list = []

    # Shuffle 10 times, seems useless for linear regression
    for i in range(10):
        trainData_shuffle, trainLabel_shuffle = util.shuffle(trainData, trainLabel)
        clf.fit(trainData_shuffle, trainLabel_shuffle)
        acc_i = clf.score(testData, testLabel)
        print("%d acc: " % i, acc_i)
        acc_list.append(acc_i)

    acc = np.mean(np.array(acc_list))
    print("Linear Regression accuracy: ", acc)
    return acc