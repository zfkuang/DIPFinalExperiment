from __future__ import print_function
import numpy as np
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import layer
import pdb

n_epochs = 100
n_episodes = 100
n_way = 100
n_shot = 5
n_query = 25
im_width, im_height, channels = 227, 227, 3
n_features = 4096
h_dim = 32
z_dim = 32
output_dim = 500
n_test_episodes = 100
n_test_way = 50
n_test_example = 10
n_test_shot = 10
n_test_query = 50

n_classes = 1000
n_test_classes = 50
n_train_classes = 700

loss_lambda = 0
learning_rate = 0.001


def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv

def encoder(x, output_dim, keep_prob, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = tf.layers.dense(x, 1024, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.dense(net, 500, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.dense(net, output_dim, activation=None, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        #net = tf.layer.AlexNet(x, 0.5, 2000, ["fc8"])
        #net = conv_block(x, h_dim, name='conv_1')
        #net = conv_block(net, h_dim, name='conv_2')
        #net = conv_block(net, h_dim, name='conv_3')
        #net = conv_block(net, z_dim, name='conv_4')
        #net = tf.contrib.layers.flatten(net)
        return net #, net

def get_distance(a, b, keep_prob=1.0, reuse=False):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    # a.shape = b.shape = N x M x D

    # use euclidean_distance
    # return tf.reduce_mean(tf.square(a - b), axis=2)

    # use cosine similarity:
    a_norm = tf.norm(a, axis=2)
    b_norm = tf.norm(b, axis=2)
    norm_product = tf.multiply(a_norm, b_norm)
    dot = tf.reduce_sum(tf.multiply(a, b), axis=2)
    ones = tf.ones(shape=tf.shape(dot))
    return 10 * (tf.subtract(ones, tf.divide(dot, norm_product)))

    # use deep network
    # return distanceNetwork(a, b, keep_prob, reuse=reuse)

def distanceNetwork(a, b, keep_prob, reuse=False):
    # use deep network
    x = tf.concat([a, b], -1)
    x = tf.reshape(x, (-1, output_dim * 2))
    with tf.variable_scope('distance', reuse=reuse):
        net = tf.layers.dense(x, 300, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.dense(net, 200, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        net = tf.contrib.layers.batch_norm(net, updates_collections=None, decay=0.99, scale=True, center=True)
        net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.dense(net, 1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        return tf.reshape(net, shape=tf.shape(a)[:-1])


class prototypicalNetwork(object):
    def __init__(self, sess):
        # Load Train Dataset

        # inputData_ = tf.placeholder(tf.float32, [None, 227, 227, channels])
        # resizeData_ = tf.image.resize_images(inputData_, [im_width, im_height])
        # trainData = sess.run(resizeData_, {inputData_:trainData})
        # testData = sess.run(resizeData_, {inputData_:testData})

        self.x = tf.placeholder(tf.float32, [None, None, n_features]) # num_classes, num_support, feature
        self.q = tf.placeholder(tf.float32, [None, n_features]) # num_queries, feature
        self.keep_prob = tf.placeholder(tf.float32)
        x_shape = tf.shape(self.x)
        q_shape = tf.shape(self.q)
        num_classes, num_support = x_shape[0], x_shape[1]
        num_queries = q_shape[0]
        self.y = tf.placeholder(tf.int64, [None]) # num_queries
        y_one_hot = tf.one_hot(self.y, depth=num_classes)
        emb_x_all = encoder(tf.reshape(self.x, [num_classes * num_support, n_features]), output_dim, self.keep_prob)
        emb_dim = tf.shape(emb_x_all)[-1]
        self.emb_x = tf.reduce_mean(tf.reshape(emb_x_all, [num_classes, num_support, emb_dim]), axis=1)
        self.emb_q = encoder(self.q, output_dim, self.keep_prob, reuse=True)
        self.dists = get_distance(self.emb_q, self.emb_x, self.keep_prob)
        self.log_p_y = tf.reshape(tf.nn.log_softmax(-self.dists), [num_queries, -1])
        self.ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, self.log_p_y), axis=-1), [-1]))

        self.logits = tf.argmax(self.log_p_y, axis=-1)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.logits, self.y)))

        params = tf.trainable_variables()[-6:]
        print(params)

        # use deep network
        # ce_loss_1 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        # emb_x_tile = tf.tile(tf.reshape(emb_x, (num_classes, 1, emb_dim)), (1, num_support, 1))
        # x_dist = distanceNetwork(tf.reshape(emb_x_all, (num_classes, num_support, emb_dim)), emb_x_tile, keep_prob, reuse=True)
        # emb_x_for_y = tf.gather_nd(emb_x, tf.reshape(y, (num_queries, 1)))
        # q_dist = distanceNetwork(emb_x_for_y, emb_q, keep_prob, reuse=True)
        # dist_loss = tf.reduce_mean(x_dist) + tf.reduce_mean(q_dist)
        # ce_loss = ce_loss_1 + loss_lambda * dist_loss

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.ce_loss, var_list=params)

        sess.run(tf.global_variables_initializer())
        #with tf.variable_scope('encoder', reuse=True):
        #    alexNet.load_initial_weights(sess)
    def train(self, sess, trainData, trainLabel, trainIndex, inputData, inputLabel, testData, testLabel,
        sourceClassNumber=1000, novelClassNumber=50):

        inputData = inputData.reshape([novelClassNumber, inputData.shape[0]//novelClassNumber]+list(inputData.shape[1:]))
        inputLabel = inputLabel.reshape([novelClassNumber, inputLabel.shape[0]//novelClassNumber]+list(inputLabel.shape[1:]))

        print(inputData.shape, inputLabel.shape, trainData.shape)

        CI = np.random.permutation(sourceClassNumber)
        for ep in range(n_epochs):
            for epi in range(n_episodes):
                epi_classes = np.random.permutation(n_train_classes)[:n_way]
                support = np.zeros([n_way, n_shot, n_features], dtype=np.float32)
                query = np.zeros([n_way, n_query, n_features], dtype=np.float32)
                for i, epi_clss in enumerate(epi_classes):
                    epi_cls = CI[epi_clss]
                    selected = np.random.permutation(trainIndex[epi_cls].shape[0])[:n_shot + n_query]
                    support[i] = trainData[trainIndex[epi_cls][selected[:n_shot]]]
                    query[i] = trainData[trainIndex[epi_cls][selected[n_shot:n_shot+n_query]]]
                query = query.reshape([n_way*n_query, n_features])
                # labels in training doesn't matter at all
                labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
                labels = labels.reshape([n_way*n_query])
                _, ls, ac, logy = sess.run([self.train_op, self.ce_loss, self.acc, self.log_p_y],
                    feed_dict={self.x: support, self.q: query, self.y:labels, self.keep_prob: 0.6})
                if (epi+1) % 50 == 0:
                    #print(logy[:5])
                    print('[epoch {}/{}, episode {}/{}] => loss: {:.5f} acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, ls, ac))
            print('Testing...')
            acc_ = []
            loss_ = []
            # for epi in range(n_test_episodes):
            #     epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
            #     support = np.zeros([n_test_way, n_test_shot, n_features], dtype=np.float32)
            #     query = np.zeros([n_test_way, n_test_query, n_features], dtype=np.float32)
            #     for i, epi_clss in enumerate(epi_classes):
            #         epi_cls = CI[epi_clss+n_train_classes]
            #         selected = np.random.permutation(n_test_example)
            #         #support_selected = np.random.permutation()
            #         support[i] = trainData[trainIndex[epi_cls][selected[:n_test_shot]]]
            #         #query_selected = np.random.permutation(n_test_query)
            #         query[i] = trainData[trainIndex[epi_cls][selected[n_test_shot:n_test_shot+n_test_query]]]
            #     query = query.reshape([n_test_way*n_test_query, n_features])
            #     labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
            #     labels = labels.reshape([n_test_way*n_test_query])
            #     ac, ls, logit, logy = sess.run([acc, ce_loss, logits, log_p_y], feed_dict={x: support, q: query, y:labels, keep_prob: 1.0})
            #     if epi == 0:
            #         print(logy[:5])
            #         print(logit[:5])
            #     acc_.append(ac)
            #     loss_.append(ls)
            for epi in range(n_test_episodes):
                epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
                # support = np.zeros([n_test_way, n_test_shot, n_features], dtype=np.float32)
                # query = np.zeros([n_test_way, n_test_query, n_features], dtype=np.float32)
                # for i, epi_cls in enumerate(epi_classes):
                #     selected = np.random.permutation(n_test_example)
                #     support_selected = np.random.permutation(n_test_shot)
                #     support[i] = inputData[epi_cls, support_selected]
                #     query_selected = np.random.permutation(n_test_query)
                #     query[i] = testData[epi_cls, query_selected]
                support = inputData
                query = testData#query.reshape([n_test_way*n_test_query, n_features])
                labels = testLabel#np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
                #labels = labels.reshape([n_test_way*n_test_query])
                ac, ls, logit, logy = sess.run([self.acc, self.ce_loss, self.logits, self.log_p_y], feed_dict={self.x: support, self.q: query, self.y:labels, self.keep_prob: 1.0})
                if epi == 0:
                    pass
                    #print(logy[:5])
                    #print(logit[:5])
                acc_.append(ac)
                loss_.append(ls)
            print('Average Test Accuracy: {:.5f}, Loss: {:.5f}'.format(np.mean(acc_), np.mean(loss_)))

    def inference(self, sess, inputShot, inputQuery):
        eX, eQ = sess.run([self.emb_x, self.emb_q], {self.x:inputShot, self.q:inputQuery, self.keep_prob: 1.0})
        return eX, eQ, get_distance(eX, eQ)
