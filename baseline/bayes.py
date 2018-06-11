from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np

import util

'''
Method:
1、预处理数据：对数据进行进行归一化处理，使数据均值为0，方差为1。
2、如果数据维数很高，使用PCA等降维。
3、分割训练集，要有验证集。如果训练集样本比较少，考虑使用交叉验证。
'''

'''
TODO:
1. feature的先验概率分布？
'''

def bayes(trainData, trainLabel, testData, testLabel, **kwargs):
    print(kwargs)
    trainData = util.normalization(trainData)
    testData = util.normalization(testData)
    acc_list = []
    acc_max = 0
    ret = []
    for i in range(10):
        trainData_shuffle, trainLabel_shuffle = util.shuffle(trainData, trainLabel)
        clf = GaussianNB()
        if kwargs['PCA']:
            pca = PCA(n_components=kwargs['n_components'])
            trainData_shuffle = pca.fit_transform(trainData_shuffle)
            clf.fit(trainData_shuffle, trainLabel_shuffle)
            testData_PCA = pca.transform(testData)
            acc_i = clf.score(testData_PCA, testLabel)
            if acc_i > acc_max:
                acc_max = acc_i
                ret = clf.predict(testData_PCA)
        else:
            clf.fit(trainData_shuffle, trainLabel_shuffle)
            acc_i = clf.score(testData, testLabel)
        print("%d acc: " % i, acc_i)
        acc_list.append(acc_i)
    acc = np.mean(np.array(acc_list))
    print("Naive Bayes accuracy: ", acc)
    return ret, acc_max