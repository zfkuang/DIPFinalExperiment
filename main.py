import tensorflow as tf 
import numpy as np 
import os
import sys
import re 
import pdb 

import util
import layer


if __name__=="__main__":
	sess = tf.Session()

	trainData, trainLabel, testData, testLabel = util.uploadData(sess)

	input_placeholder=tf.placeholder(tf.float32, shape=[None,227,227,3])
	#originShape=tf.placeholder(tf.float32, shape=[None, 4096]) #63996, 4096
	net = layer.AlexNet(input_placeholder, 0.9, 1000, [])
	net.load_initial_weights(sess)

	b = sess.run(net.fc8, feed_dict={input_placeholder:trainData[0:50]})
	pdb.set_trace()