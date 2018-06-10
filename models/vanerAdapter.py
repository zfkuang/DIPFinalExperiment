import numpy as np
import util
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow


def cosine_distance(x, y):
    return 1. * np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)


def adaptVaner(sess, **kwargs):
    a_new = 