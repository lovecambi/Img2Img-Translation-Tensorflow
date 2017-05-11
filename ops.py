# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:49:02 2017

@author: fankai
"""

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import math
import scipy.misc


def ganloss(yl, c=0.99):
    """
    input is the logits
    c is soft label
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yl, labels=c*tf.ones_like(yl)))
    

def lrelu(x, leaky=0.2):
    return tf.maximum(x, leaky*x)
    
    
def batch_norm(x, train_mode=True, epsilon=1e-5, momentum=0.9, name="bn"):
    with tf.variable_scope(name):
        return tcl.batch_norm(x, 
                              decay=momentum, 
                              updates_collections=None, 
                              epsilon=epsilon, 
                              scale=True, 
                              is_training=train_mode, 
                              trainable=True, 
                              scope=name)


def conv2d(input_, output_dim, k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02, bias=False, name="conv2d"):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, W, strides=[1, stride_h, stride_w, 1], padding='SAME')
        if bias:
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = conv + b
        return conv


def deconv2d(x, output_shape, 
             k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02,
             bias=True, padding='SAME', name='deconv2d'):
    in_C = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable('W', [k_h, k_w, output_shape[-1], in_C], initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable('b', [output_shape[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
            deconv = deconv + b
    return deconv


def iterate_minibatches_u(datasize, batchsize, shuffle=False):
    """
    This function tries to iterate unlabeled data in mini-batch
    for batch_data in iterate_minibatches_u(data, batchsize, True):    
        #processing batch_data
    """
    if shuffle:
        indices = np.arange(datasize)
        np.random.RandomState(np.random.randint(1,2147462579)).shuffle(indices)
    for start_idx in xrange(0, datasize - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt
        

def toshow(x):
    bs = x.shape[0]
    n = int(math.sqrt(bs))
    assert(n*n == bs)
    bs, H, W, C = x.shape
    x = np.reshape(x, [n, n, H, W, C])
    x = np.transpose(x, [1, 2, 0, 3, 4])
    return x.reshape([-1, n*W, C]) if C > 1 else x.reshape([-1, n*W])
        