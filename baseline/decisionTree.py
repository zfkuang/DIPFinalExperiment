# -*- encoding: utf-8 -*

from sklearn.tree import DecisionTreeClassifier
import numpy as np

import util

'''
Method:
1、预处理数据：对数据进行进行归一化处理，使数据均值为0，方差为1。
2、分割训练集，要有验证集。如果训练集样本比较少，考虑使用交叉验证。
'''

def decisionTree(trainData, trainLabel, testData, testLabel, **kwargs):
    print(kwargs)
    trainData = util.normalization(trainData)
    testData = util.normalization(testData)

    shuffle_times = 10
    acc_list = []

    for i in range(shuffle_times):
        trainData_shuffle, trainLabel_shuffle = util.shuffle(trainData, trainLabel)
        cdt = DecisionTreeClassifier()
        cdt.fit(trainData_shuffle, trainLabel_shuffle)
        now_acc = cdt.score(testData, testLabel)
        acc_list.append(now_acc)
        print("%d acc, %.3f" % (i, now_acc))
    acc = np.mean(np.array(acc_list))
    print("Decision Tree accuracy: ", acc)
    return acc