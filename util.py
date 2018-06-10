import tensorflow as tf
import numpy as np
import pdb
import os
import sys
import re
from sklearn.preprocessing import normalize
import layer

def imageFileToArray(session, filename):
        # load and preprocess the image
    with tf.variable_scope("image", reuse=True):
        VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        return session.run(img_bgr)

def extractFeature(sess, model, data):
    return sess.run([model.fc7], feed_dict={model.X:data})[0]

def uploadData(sess, sampleNumber, dataFolder, fileNameRegex, groupInFilename):

    inputData, inputLabel = None, None

    if os.path.exists("data//"+dataFolder) != True:
        return None, None

    if groupInFilename != True:
        print("Label info is in a file, reading label file...")
        inputLabel = np.load("data//testingLabel.npy")
        inputLabel = inputLabel-np.min(inputLabel)
    else:
        inputLabel = np.zeros((500), dtype=np.uint32)
        for i in range(50):
            for j in range(10):
                inputLabel[i*10+j] = i

    npFileName = "data//"+dataFolder+"Data.npy"
    if os.path.exists(npFileName) != True:
        print("Creating new datafile of dataset: "+dataFolder)
        inputData = np.zeros((sampleNumber, 4096), dtype=np.float32)
        batchInputData = np.zeros((300, 227, 227, 3), dtype=np.float32)
        batchCnt = 0
        totalCnt = 0
        ind = np.zeros(sampleNumber, dtype=np.int32)
        imageName = fileNameRegex
        for dirPath, dirNames, fileNames in os.walk("data//"+dataFolder):
            #print(dirPath, dirNames, fileNames)
            for fileName in fileNames:
                matchResult = re.match(imageName, fileName)
                if matchResult != None:
                    temp = matchResult.groupdict()
                    if groupInFilename == True:
                        group = int(temp['group'])-1
                        index = int(temp['index'])-1
                        ind[totalCnt] = group*10+index
                        print(group, index)
                    else:
                        index = int(temp['index'])-1
                        ind[totalCnt] = index
                        print(index)
                    batchInputData[batchCnt] = imageFileToArray(sess, dirPath+"//"+fileName)
                    batchCnt = batchCnt+1
                    totalCnt = totalCnt+1
                    if batchCnt==300 or totalCnt==sampleNumber:
                        data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
                        model = layer.AlexNet(data_, 1, 1000, [])
                        model.load_initial_weights(sess)
                        inputData[totalCnt-batchCnt:totalCnt] = extractFeature(sess, model, batchInputData)[:batchCnt]
                        tf.reset_default_graph()
                        batchCnt = 0
        iind = np.copy(ind)
        for i in range(ind.shape[0]):
            iind[ind[i]] = i
        inputData = inputData[iind]
        np.save(npFileName, inputData)
    else:
        print("Loading datafile of dataset: "+dataFolder)
        inputData = np.load(npFileName)
        print(inputData.shape)

    #pdb.set_trace()
    return inputData, inputLabel

def uploadBasicData():
    inputData, inputLabel = None, None

    if os.path.exists("data//fc7.npy") != True or os.path.exists("data//base_classes.txt") != True:
        return None, None
    if os.path.exists("data//newLabel.npy") != True and (os.path.exists("data//label.npy") != True or os.path.exists("data//correct.txt") != True):
        return None, None

    print("Loading base classes data...")

    if(os.path.exists("data//newLabel.npy") != True):
        sourceLabel = np.load("data//label.npy")
        correctFile = open("data//correct.txt", "r")
        correct = correctFile.read().split("\n")[:-1]
        for i in range(len(correct)):
            correct[i] = int(correct[i].split(' ')[1])-1
        correct = np.array(correct)
        inputLabel = correct[sourceLabel]
        #for i in range(len(sourceLabel)):
        #    sourceLabel[i] = correct[sourceLabel[i]]
        np.save("data//newLabel.npy", inputLabel)
    else:
        inputLabel = np.load("data//newLabel.npy")

    index = [[] for i in range(1000)]
    for i in range(inputLabel.shape[0]):
        index[inputLabel[i]].append(i)
    for i in range(1000):
        index[i] = np.array(index[i])

    inputData = np.load("data//fc7.npy")

    return inputData, inputLabel, index

def divideData(inputData, inputLabel, classNumber=50, trainSample=7, testSample=3):

    shuffle0 = list(range(50))
    shuffle1 = list(range(10))
    #np.random.shuffle(shuffle0)
    #np.random.shuffle(shuffle1)

    inputData = inputData[shuffle0]
    inputLabel = inputLabel[shuffle0]
    inputData = inputData[:,shuffle1]
#    inputData = np.sort(inputData, axis=0, order=shuffle_list)
#    inputLabel = np.sort(inputLabel, axis=0, order=shuffle_list)
    trainData = inputData[:classNumber, :trainSample]
    testData = inputData[:classNumber, trainSample:trainSample+testSample]
    trainLabel = inputLabel[:classNumber, :trainSample]
    testLabel = inputLabel[:classNumber, trainSample:trainSample+testSample]
    print(testData.shape, trainData.shape)
    print(testLabel.shape, trainLabel.shape)
    trainData = trainData.reshape([classNumber*trainSample, 227, 227, 3])
    testData = testData.reshape([classNumber*testSample, 227, 227, 3])
    trainLabel = trainLabel.reshape([classNumber*trainSample])
    testLabel = testLabel.reshape([classNumber*testSample])

    return trainData, trainLabel, testData, testLabel

def shuffle(data, label):
    size = data.shape[0]
    shuffle_list = list(range(size))

    np.random.shuffle(shuffle_list)
    data = data[shuffle_list]
    label = label[shuffle_list]

    return data, label

def normalization(data):
    originShape = data.shape
    data = data.reshape((data.shape[0], np.multiply.reduce(data.shape[1:])))
    data = normalize(data)
    return data.reshape(originShape)

def loadBaseClassifier():
    param_dicts = np.load("data/base_classifier.npy").item()
    classifier = [[]] * len(param_dicts.keys())
    for name, param_dict in param_dicts.items():
        param_dict = param_dict.item()
        kernel = param_dict['kernel']
        bias = param_dict['bias']
        bias = bias.reshape((1, -1))
        data = np.concatenate((kernel, bias))
        data = np.reshape(data, (np.multiply.reduce(data.shape)))
        classifier[int(name.split('_')[-1])] = data
    return np.array(classifier)

def loadNovelClassifier():
    param_dicts = np.load("data/novel_classifier.npy").item()
    classifier = [[]] * len(param_dicts.keys())
    for name, param_dict in param_dicts.items():
        param_dict = param_dict.item()
        kernel = param_dict['kernel']
        bias = param_dict['bias']
        bias = bias.reshape((1, -1))
        data = np.concatenate((kernel, bias))
        data = np.reshape(data, (np.multiply.reduce(data.shape)))
        classifier[int(name.split('_')[-1])] = data
    return np.array(classifier)
