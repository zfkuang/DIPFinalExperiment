from sklearn.neighbors import KNeighborsClassifier
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
        trainData_shuffle, trainLabel_shuffle = util.shuffle(trainData, trainLabel)
        if kwargs['PCA']:
            pca = PCA(n_components=100)
            trainData_shuffle = pca.fit_transform(trainData_shuffle)
            neigh.fit(trainData_shuffle, trainLabel_shuffle)
            testData_PCA = pca.transform(testData)
            acc_i = neigh.score(testData_PCA, testLabel)
        else:
            neigh.fit(trainData_shuffle, trainLabel_shuffle)
            acc_i = neigh.score(testData, testLabel)
        print("%d acc: " % i, acc_i)
        acc_list.append(acc_i)
    acc = np.mean(np.array(acc_list))
    print("KNN accuracy: ", acc)
    return acc