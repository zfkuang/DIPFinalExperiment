import numpy as np
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import layer
import pdb
import util
from tensorflow.python import pywrap_tensorflow

def encoder(x, output_dim, keep_prob, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        # net = tf.layers.dense(x, output_dim, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        # net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        # net = tf.nn.dropout(net, keep_prob)
        # net = tf.layers.dense(net, 500, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        # net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        # net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.dense(x, output_dim, activation=None, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        return net


class binary_classifier(object):
    def __init__(self, sess, **kwargs):

        self.data_ = tf.placeholder(tf.float32, shape=[None, 4096])
        self.label_ = tf.placeholder(tf.int32, shape=[None])
        self.keep_prob_ = tf.placeholder(tf.float32)

        if kwargs['model'] == 'mlp':
            self.logits = encoder(self.data_, 2, self.keep_prob_)
        else:
            return None

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_, logits=self.logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.label_)
        self.pred = tf.argmax(self.logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(kwargs['learning_rate']), trainable=False, dtype=tf.float32)
        self.learning_rate_decay = self.learning_rate.assign(self.learning_rate * kwargs['learning_rate_decay'])

        self.global_step = tf.Variable(0, trainable=False)

        #Very ugly code, we directly choose variables that should be trained,
        #since setting them as not trainable in reuse mode has no effect at all.
        self.params = tf.trainable_variables()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)

        # self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                #    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        sess.run(tf.global_variables_initializer())

        if kwargs['test'] == True:
            self.load_initial_weights(sess, **kwargs)

    def load_initial_weights(self, sess, **kwargs):
        weights_dict = np.load(kwargs['weight_path'], encoding='bytes').item()
        for op_name in weights_dict:
            print("loading ", op_name)
            with tf.variable_scope('encoder', reuse=True):
                data = weights_dict[op_name]
                # Biases
                if len(data.shape) == 1:
                    var = tf.get_variable('dense/bias', trainable=False)
                    sess.run(var.assign(data))
                # Weights
                else:
                    var = tf.get_variable('dense/kernel', trainable=False)
                    sess.run(var.assign(data))

    def train_epoch(self, sess, data, label, **kwargs):
        loss, acc = 0.0, 0.0
        st, ed, times = 0, kwargs['batch_size'], 0
        while st < len(data) and ed <= len(data):
            data_batch, label_batch = data[st:ed], label[st:ed]
            loss_, acc_, _ = sess.run([self.loss, self.acc, self.train_op], feed_dict={self.data_:data_batch, self.label_:label_batch, self.keep_prob_:kwargs['keep_prob']})
            loss += loss_
            acc += acc_
            st, ed = ed, ed+kwargs['batch_size']
            times += 1
        loss /= times
        acc /= times

        # print( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))
        # print(tf.global_variables())

        return loss, acc

    def test(self, sess, data, label):
        return sess.run([self.loss, self.acc], feed_dict={self.data_:data, self.label_:label, self.keep_prob_:1.0})

    def inference(self, sess, data):
        return sess.run([self.pred], feed_dict={self.data_:data, self.keep_prob_:1.0})

    def save_model(self, sess, **kwargs):
        saver = tf.train.Saver()
        if not os.path.exists("data/save_model"):
            os.mkdir("data/save_model")
        modelPath = "data/save_model/base_class_%d/" % kwargs['trainNum']
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)
        checkpoint_path = modelPath + "save.ckpt"
        saver.save(sess, checkpoint_path)
        reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map=reader.get_variable_to_shape_map()

        for key in var_to_shape_map:
            str_name = key
            if str_name.find('Adam') > -1:
                continue
            if (str_name == 'encoder/dense/kernel'):
                kernel = reader.get_tensor(key)
            if (str_name == 'encoder/dense/bias'):
                bias = reader.get_tensor(key)
        network = {"kernel":kernel, "bias":bias}
        np.save(modelPath + "save.npy", network)


def classify(sess, trainData, trainLabel, testData, testLabel, **kwargs):
    net = binary_classifier(sess, **kwargs)

    trainData, trainLabel = util.shuffle(trainData, trainLabel)
    print(kwargs['trainNum'], "train", net.train_epoch(sess, trainData, trainLabel, **kwargs))
    # print(kwargs['trainNum'], "test", net.test(sess, testData, testLabel))
    # print(net.inference(sess, testData))
    net.save_model(sess, **kwargs)


def train_base_classifier(sess, trainData, trainLabel, trainIndex, **kwargs):
    for i, indexlist in enumerate(trainIndex):
        pos_data = trainData[indexlist]
        pos_label = [1] * len(indexlist)
        templist = list(np.arange(0, len(trainData)))
        for l in indexlist:
            templist.remove(l)
        neg_data = trainData[templist]
        neg_label = [0] * len(templist)
        data = np.concatenate((pos_data, neg_data))
        label = np.concatenate((pos_label, neg_label))
        data, label = util.shuffle(data, label)
        classify(sess, data, label, None, None, trainNum=i, test=False, **kwargs)
        tf.get_variable_scope().reuse_variables()

def test_base_classifier(sess, testData, testLabel, **kwargs):
    net = binary_classifier(sess, test=True, **kwargs)
    print(net.test(sess, testData, testLabel))