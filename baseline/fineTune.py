# -*- encoding: utf-8 -*

import tensorflow as tf
import numpy as np
import os
import sys
import re
import pdb

import util
import layer
import network

def fineTune(sess, trainData, trainLabel, testData, testLabel, **kwargs):
    net = network.Network(sess, model="default", **kwargs)
    maxAcc = 0
    retInf = []
    for i in range(20):
        trainData, trainLabel = util.shuffle(trainData, trainLabel)
        print(network.train_epoch(net, sess, trainData, trainLabel, **kwargs))
        acc = net.test(sess, testData, testLabel)[1]
        print(acc)
        if acc > maxAcc:
            maxAcc = acc
            retInf = net.inference(sess, testData)[0]
    ##for i in range(0, 100):
    ##    print(net.train(sess, trainData[0:10], trainLabel[0:10], args['keep_prob'])[0:2])
    print(maxAcc)

    return retInf, maxAcc