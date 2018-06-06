from __future__ import print_function
import numpy as np
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import layer
import pdb

def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv

def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = layer.AlexNet(x, 0.5, 2000, ["fc8"])
        #net = conv_block(x, h_dim, name='conv_1')
        #net = conv_block(net, h_dim, name='conv_2')
        #net = conv_block(net, h_dim, name='conv_3')
        #net = conv_block(net, z_dim, name='conv_4')
        # net = tf.contrib.layers.flatten(net)
        return net.fc8, net

def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

n_epochs = 20
n_episodes = 100
n_way = 30
n_shot = 1
n_query = 6
n_examples = 7
im_width, im_height, channels = 227, 227, 3
h_dim = 32
z_dim = 32
n_test_episodes = 1000
n_test_way = 50
n_test_shot = 7
n_test_query = 3

def prototypicalNetwork(sess, trainData, trainLabel, testData, testLabel, classNumber=50, **kwargs):
    # Load Train Dataset

    # inputData_ = tf.placeholder(tf.float32, [None, 227, 227, channels])
    # resizeData_ = tf.image.resize_images(inputData_, [im_width, im_height])
    # trainData = sess.run(resizeData_, {inputData_:trainData})
    # testData = sess.run(resizeData_, {inputData_:testData})

    trainData = trainData.reshape([classNumber, trainData.shape[0]//classNumber]+list(trainData.shape[1:]))
    trainLabel = trainLabel.reshape([classNumber, trainLabel.shape[0]//classNumber]+list(trainLabel.shape[1:]))
    testData = testData.reshape([classNumber, testData.shape[0]//classNumber]+list(testData.shape[1:]))
    testLabel = testLabel.reshape([classNumber, testLabel.shape[0]//classNumber]+list(testLabel.shape[1:]))
    n_classes = trainData.shape[0]
    n_test_classes = trainData.shape[0]
    print(trainData.shape)

    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels]) # num_classes, num_support, imh, imw, chan
    q = tf.placeholder(tf.float32, [None, im_height, im_width, channels]) # num_queries, imh, imw, chan
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    num_classes, num_support = x_shape[0], x_shape[1]
    num_queries = q_shape[0]
    y = tf.placeholder(tf.int64, [None]) # num_queries
    y_one_hot = tf.one_hot(y, depth=num_classes)
    emb_x, alexNet = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
    emb_dim = tf.shape(emb_x)[-1]
    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)
    emb_q, _ = encoder(q, h_dim, z_dim, reuse=True)
    dists = euclidean_distance(emb_q, emb_x)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_queries, -1])
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

    params = tf.trainable_variables()[-2:]
    print(params)
    train_op = tf.train.AdamOptimizer().minimize(ce_loss, var_list=params)

    sess.run(tf.global_variables_initializer())
    with tf.variable_scope('encoder', reuse=True):
        alexNet.load_initial_weights(sess)


    for ep in range(n_epochs):
        for epi in range(n_episodes):
            epi_classes = np.random.permutation(n_classes)[:n_way]
            support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)
            query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(n_examples)[:n_shot + n_query]
                support[i] = trainData[epi_cls, selected[:n_shot]]
                query[i] = trainData[epi_cls, selected[n_shot:]]
            query = query.reshape([n_way*n_query, im_height, im_width, channels])
            # labels in training doesn't matter at all
            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
            labels = labels.reshape([n_way*n_query])
            _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y:labels})
            if (epi+1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, ls, ac))

        epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
        support = np.zeros([n_test_way, n_test_shot, im_height, im_width, channels], dtype=np.float32)
        query = np.zeros([n_test_way, n_test_query, im_height, im_width, channels], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes):
            support_selected = np.random.permutation(n_test_shot)
            support[i] = trainData[epi_cls, support_selected]
            query_selected = np.random.permutation(n_test_query)
            query[i] = testData[epi_cls, query_selected]
        query = query.reshape([n_test_way*n_test_query, im_height, im_width, channels])
        labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
        labels = labels.reshape([n_test_way*n_test_query])
        ac, ls = sess.run([acc, ce_loss], feed_dict={x: support, q: query, y:labels})
        print('Average Test Accuracy: {:.5f}, Loss: {:.5f}'.format(ac, ls))    
    print('Testing...')