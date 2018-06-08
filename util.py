import tensorflow as tf
import numpy as np
import pdb
import os
import sys
import re
from sklearn.preprocessing import normalize


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

def uploadData(sess):

    inputData = None

    if os.path.exists("data//training") != True:
        return None, None

    if os.path.exists("data//inputData.npy") != True:
        print("Creating new input array...")
        inputData = np.zeros((50, 10, 227, 227, 3), dtype=np.float32)
        imageName = r"(?P<group>\d{3})_(?P<index>\d{4}).jpg"
        folder = "data//training"
        for dirPath, dirNames, fileNames in os.walk(folder):
            #print(dirPath, dirNames, fileNames)
            for fileName in fileNames:
                matchResult = re.match(imageName, fileName)
                if matchResult != None:
                    temp = matchResult.groupdict()
                    #print(matchResult.groupdict())
                    print(temp['group'], temp['index'])
                    group = int(temp['group'])
                    index = int(temp['index'])
                    inputData[group-1][index-1] = imageFileToArray(sess, dirPath+"//"+fileName)
        np.save("data//inputData.npy", inputData)
    else:
        print("Loading input array file...")
        inputData = np.load("data//inputData.npy")
        print(inputData.shape)

    print("Generating label...")
    inputLabel = np.zeros((50, 10), dtype=np.int32)
    for group in range(0, 50):
        for index in range(0, 10):
            inputLabel[group, index] = group

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

def extractFeature(sess, model, data):
    return sess.run([model.fc7], feed_dict={model.X:data})[0]