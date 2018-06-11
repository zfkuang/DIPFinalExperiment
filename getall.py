import os
import numpy as np
import layer
import tensorflow as tf
import util

# files = os.listdir('data/save_model')
# alist = list(range(0,50))
# res = {}
# for filename in files:
#     alist.remove(int(filename.split('_')[-1]))
#     a = np.load('data/save_model/'+filename+'/save.npy')
#     res[filename] = a

# print(alist)

# np.save('data/novel_classifier.npy', res)


###########


# sess = tf.Session()
# data_ = tf.placeholder(tf.float32, shape=[None,227,227,3])
# keep_prob_ = tf.placeholder(tf.float32)
# net = layer.AlexNet(data_, keep_prob_, 1000, [])
# sess.run(tf.global_variables_initializer())
# net.load_initial_weights(sess)


a=np.load("data/em_basic.npy")
print(a.shape)
a = a.reshape(1000,-1)
np.save("data/em_basic.npy", a)