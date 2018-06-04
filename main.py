import tensorflow as tf 
import numpy as np 
import os
import sys
import re 
import pdb 

import util
import layer
import network

### Algorithm: fine-tune pre-trained model
## usage: fineTune(sess, trainData, trainLabel, testData, testLabel, **kwargs)
import baseline.fineTune
fineTuneArgs = {
	"batch_size":5,
	"keep_prob":0.4,
	"learning_rate":0.001,
	"learning_rate_decay":0.999
}

if __name__=="__main__":

	# Initialization
	sess = tf.Session()
	trainData, trainLabel, testData, testLabel = util.uploadData(sess)

	# Training & Testing
	fineTuneAcc = baseline.fineTune.fineTune(sess, trainData, trainLabel, testData, testLabel, **fineTuneArgs)

	pdb.set_trace()