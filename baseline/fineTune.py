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

	for i in range(10):
		trainData, trainLabel = util.shuffle(trainData, trainLabel)
		print(network.train_epoch(net, sess, trainData, trainLabel, **kwargs))
		print(net.test(sess, testData, testLabel))
	##for i in range(0, 100):
	##	print(net.train(sess, trainData[0:10], trainLabel[0:10], args['keep_prob'])[0:2])
	print(net.test(sess, testData, testLabel))
	print(net.inference(sess, testData))

	return net.test(sess, testData, testLabel)[1]