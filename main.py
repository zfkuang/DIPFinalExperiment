import tensorflow as tf 
import numpy as np 
import os
import sys
import re 
import pdb 

import util
import layer
import network

args = {
	"network":"default",
	"batch_size":10,
	"keep_prob":0.9,
	"learning_rate":0.01,
	"learning_rate_decay":0.999
}

if __name__=="__main__":
	sess = tf.Session()

	trainData, trainLabel, testData, testLabel = util.uploadData(sess)

	net = network.Network(sess, **args)

	net.train(sess, trainData[0:50], trainLabel[0:50], args['keep_prob'])
	net.train(sess, trainData[0:50], trainLabel[0:50], args['keep_prob'])
	net.train(sess, trainData[0:50], trainLabel[0:50], args['keep_prob'])
	net.train(sess, trainData[0:50], trainLabel[0:50], args['keep_prob'])
	print(net.test(sess, trainData[0:50], trainLabel[0:50]))
	print(net.inference(sess, trainData[0:50]))

	pdb.set_trace()