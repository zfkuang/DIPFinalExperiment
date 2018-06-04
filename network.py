import tensorflow as tf 
import numpy as np 
import os
import sys
import re 
import pdb 

import util
import layer

class Network(object):
	def __init__(self, sess, **kwargs):

		self.data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
		self.label_ = tf.placeholder(tf.int32, shape=[None])
		self.keep_prob_ = tf.placeholder(tf.float32)

		if kwargs['network'] == 'default':
			self.net = layer.AlexNet(self.data_, self.keep_prob_, 50, ['fc8'])
			self.net.load_initial_weights(sess)
			logits = self.net.fc8
		else:
			return None

		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_, logits=logits))
		self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.label_)
		self.pred = tf.argmax(logits, 1)
		self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		self.learning_rate = tf.Variable(float(kwargs['learning_rate']), trainable=False, dtype=tf.float32)
		self.learning_rate_decay = self.learning_rate.assign(self.learning_rate * kwargs['learning_rate_decay'])

		self.global_step = tf.Variable(0, trainable=False)
		self.params = tf.trainable_variables()
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)

		self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
									max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

	def train(self, sess, data, label, keep_prob):
		return sess.run(self.train_op, feed_dict={self.data_:data, self.label_:label, self.keep_prob_:keep_prob})

	def test(self, sess, data, label):
		return sess.run([self.loss, self.acc], feed_dict={self.data_:data, self.label_:label, self.keep_prob_:1.0})

	def inference(self, sess, data):
		return sess.run([self.pred], feed_dict={self.data_:data, self.keep_prob_:1.0})



