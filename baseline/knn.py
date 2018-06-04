from sklearn.neighbors import KNeighborsClassifier
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
1. n_neighbors can be learnt by word-vector.
2. weights can use 'uniform', 'distance' or other [callable] function to compute weights.
3. p can use 1 or 2 or any for minkowski distance.
'''

def knn(trainData, trainLabel, testData, testLabel, **kwargs):
    print(kwargs)
    neigh = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'], weights=kwargs['weights'], p=kwargs['p'])
    trainData = util.normalization(trainData.reshape((trainData.shape[0], trainData.shape[1] * trainData.shape[2] * trainData.shape[3])))
    testData = util.normalization(testData.reshape((testData.shape[0], testData.shape[1] * testData.shape[2] * testData.shape[3])))
    acc_list = []
    for i in range(10):
        trainData, trainLabel = util.shuffle(trainData, trainLabel)
        neigh.fit(trainData, trainLabel)
        acc_list.append(neigh.score(testData, testLabel))
    acc = np.mean(np.array(acc_list))
    print("KNN accuracy: ", acc)
    return acc