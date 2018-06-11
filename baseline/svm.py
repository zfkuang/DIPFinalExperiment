from sklearn.svm import SVC, LinearSVC
import numpy as np

import util


def svm(trainData, trainLabel, testData, testLabel, **kwargs):
    print(kwargs)
    #neigh = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'], weights=kwargs['weights'], p=kwargs['p'])
    #trainData = util.normalization(trainData.reshape((trainData.shape[0], trainData.shape[1] * trainData.shape[2] * trainData.shape[3])))
    #testData = util.normalization(testData.reshape((testData.shape[0], testData.shape[1] * testData.shape[2] * testData.shape[3])))
    linearSVC_clf = LinearSVC()
    #this
    acc_list = []
    acc_max = 0
    ret = []
    shuffleTimes = 1
    for i in range(shuffleTimes):
        print(i+1, '/', shuffleTimes)
        trainData, trainLabel = util.shuffle(trainData, trainLabel)
        linearSVC_clf.fit(trainData, trainLabel)
        acc_i = linearSVC_clf.score(testData, testLabel)
        acc_list.append(acc_i)
        print("LinearSVC accuracy: ", np.mean(np.array(acc_list)))
        if acc_i > acc_max:
            acc_max = acc_i
            ret = linearSVC_clf.predict(testData)
    acc = np.mean(np.array(acc_list))
    print("LinearSVC accuracy: ", acc)

    return ret, acc_max

    SVC_clf = SVC()
    acc_list = []
    shuffleTimes = 50
    for i in range(shuffleTimes):
        print(i+1, '/', shuffleTimes)
        trainData, trainLabel = util.shuffle(trainData, trainLabel)
        SVC_clf.fit(trainData, trainLabel)
        acc_list.append(SVC_clf.score(testData, testLabel))
        print("SVC accuracy: ", np.mean(np.array(acc_list)))
    acc = np.mean(np.array(acc_list))
    print("SVC accuracy: ", acc)
