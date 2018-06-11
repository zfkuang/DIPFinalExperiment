import tensorflow as tf
import numpy as np
import pdb
import os
import sys
import re
from sklearn.preprocessing import normalize
import layer
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image


# def imageDataAugmentation():
#     datagen = ImageDataGenerator(
#             rotation_range=40,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True,
#             vertical_flip=True,
#             fill_mode='nearest')

#     img = load_img('data/training/005.baseball-glove/005_0001.jpg')  # 这是一个PIL图像
#     print(img)
#     x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
#     x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

#     # 下面是生产图片的代码
#     # 生产的所有图片保存在 `preview/` 目录下
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#         i += 1
#         if i > 50:
#             break  # 否则生成器会退出循环

def imageFileToArray(session, filename):
        # load and preprocess the image
    with tf.variable_scope("image", reuse=True):
        VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        return session.run(img_resized)

seq = iaa.SomeOf(3, [
    iaa.Crop(px=(0, 10)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    iaa.Add((-20, 20), per_channel=0),
    # iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.ContrastNormalization((0.8, 1.2)),
    iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(-5, 5), shear=(-7, 7),  mode=["edge"])
])
def transImage(images, withAugmentation):
    if withAugmentation:
        images = seq.augment_images(images)
    images = images-[123.68, 116.779, 103.939]
    images = images[:, :, :, ::-1]
    return images

def extractFeature(sess, model, data):
    return sess.run([model.fc7], feed_dict={model.X:data})[0]

def uploadData(dataAugment, sampleNumber, dataFolder, fileNameRegex, groupInFilename):

    sess = tf.Session()
    inputData, inputLabel = None, None

    if os.path.exists("data//"+dataFolder) != True:
        return None, None

    if groupInFilename != True:
        print("Label info is in a file, reading label file...")
        inputLabel = np.load("data//testingLabel.npy")
        inputLabel = inputLabel-np.min(inputLabel)
    else:
        inputLabel = np.zeros((500*(dataAugment+1)), dtype=np.uint32)
        for t in range(dataAugment+1):
            for i in range(50):
                for j in range(10):
                    inputLabel[t*500+i*10+j] = i

    npFileName = "data//"+dataFolder+"Data"+str(dataAugment)+".npy"
    if os.path.exists(npFileName) != True:
        print("Creating new datafile of dataset: "+dataFolder)
        generageSampleNumber = sampleNumber*(dataAugment+1)
        inputData = np.zeros((generageSampleNumber, 4096), dtype=np.float32)
        batchInputData = np.zeros((300, 227, 227, 3), dtype=np.float32)
        batchCnt = 0
        totalCnt = 0
        fileCnt = 0
        ind = np.zeros(generageSampleNumber, dtype=np.int32)
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
                    fileCnt = fileCnt+1
                    if batchCnt==300 or fileCnt==sampleNumber:
                        data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
                        model = layer.AlexNet(data_, 1, 1000, [])
                        model.load_initial_weights(sess)
                        
                        sourceTot = totalCnt
                        inputData[totalCnt-batchCnt:totalCnt] = extractFeature(sess, model, transImage(batchInputData, 0))[:batchCnt]
                        #pdb.set_trace()
                        for i in range(dataAugment):
                            ind[totalCnt:totalCnt+batchCnt] = ind[sourceTot-batchCnt:sourceTot]+sampleNumber*(i+1)
                            inputData[totalCnt:totalCnt+batchCnt] = extractFeature(sess, model, transImage(batchInputData, 1))[:batchCnt]
                            totalCnt = totalCnt+batchCnt 
                        sess.close()
                        tf.reset_default_graph()
                        sess = tf.Session()
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
    sess.close()
    tf.reset_default_graph()

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

    
def loadWnew():
    param_dicts = np.load("data/W_new.npy").item()
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
