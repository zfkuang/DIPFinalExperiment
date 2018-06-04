import tensorflow as tf
import numpy as np
import pdb
import os
import sys
import re
from sklearn.preprocessing import normalize

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def imageFileToArray(session, filename):
        # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, VGG_MEAN)

    # pdb.set_trace()
    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]
    return session.run(img_bgr)

# upload both train data and test data
# return: trainData, trainLabel, testData, testLabel
def uploadData(sess):
    trainData, testData = None, None

    if os.path.exists("training") != True:
        return None, None, None, None

    if os.path.exists("data//trainData.npy") != True or os.path.exists("data//testData.npy") != True:
        print("Creating new input array...")
        inputArray = np.zeros((50, 10, 227, 227, 3), dtype=np.float32)
        imageName = r"(?P<group>\d{3})_(?P<index>\d{4}).jpg"
        folder = "training"
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
                    inputArray[group-1][index-1] = imageFileToArray(sess, dirPath+"//"+fileName)
        for i in range(50):
            np.random.shuffle(inputArray[i])
        testData = inputArray[:, 7:]
        trainData = inputArray[:, :7]
        print(testData.shape, trainData.shape)
        trainData = trainData.reshape([50*7, 227, 227, 3])
        np.save("data//trainData.npy", trainData)
        testData = testData.reshape([50*3, 227, 227, 3])
        np.save("data//testData.npy", testData)
    else:
        print("Loading input array file...")
        trainData = np.load("data//trainData.npy")
        testData  = np.load("data//testData.npy")
        print(testData.shape, trainData.shape)

    print("Generating label...")
    testLabel = np.zeros((50*3), dtype=np.int32)
    for group in range(0, 50):
        for index in range(0, 3):
            testLabel[group*3+index] = group
    trainLabel = np.zeros((50*7), dtype=np.int32)
    for group in range(0, 50):
        for index in range(0, 7):
            trainLabel[group*7+index] = group

    return trainData, trainLabel, testData, testLabel


def shuffle(data, label):
    size = data.shape[0]
    shuffle_list = list(range(size))

    np.random.shuffle(shuffle_list)
    temp_data = np.copy(data)
    temp_label = np.copy(label)

    for i in range(size):
        temp_data[i] = data[shuffle_list[i]]
        temp_label[i] = label[shuffle_list[i]]

    return temp_data, temp_label

def normalization(data):
    originShape = data.shape 
    data = data.reshape((data.shape[0], np.multiply.reduce(data.shape[1:])))
    data = normalize(data)
    return data.reshape(originShape)

def extractFeature(sess, model, data):
    return sess.run([model.fc6], feed_dict={model.X:data})[0]