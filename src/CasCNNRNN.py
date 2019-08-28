import numpy as np
import tensorflow as tf
from Net import *

def CasCNN(x, conv_filter_shape, 
           regularizer='L2', weight_decay=0.0001, fc_layer=True, fc_n_hidden=1024, reuse=True, **kwargs):
    """
    Return a tf CNN network place holder
    Arguments:
       x: tf place holder
       conv_filter_shape: list of dictionary


        example input:
        conv_filter_shape = [{'W': [3,3,1,32], 'b': [32]}, 
                             {'W': [3,3,32,64], 'b': [64]},
                             {'W': [3,3,64,128], 'b': [128]}]
    """
    if reuse:
        net = ConvNet(x, filter_shape=conv_filter_shape, 
                      regularizer='L2', weight_decay=weight_decay, 
                      scope='Conv2D',
                      reuse=tf.AUTO_REUSE)
    else:
        net = ConvNet(x, filter_shape=conv_filter_shape, 
                      regularizer='L2', weight_decay=weight_decay)
        
        
    shape_list = net.get_shape().as_list()[1:]
    shape = 1
    for i in shape_list:
        shape *= i
    net = tf.reshape(net, [-1, shape])
    if fc_layer:
        if reuse:
            net = fullyconnected(net, W_shape=[shape, fc_n_hidden], b_shape=[fc_n_hidden], 
                                 regularizer=regularizer, weight_decay=weight_decay, 
                                 scope='casCNN_FC',
                                 reuse=tf.AUTO_REUSE)
        else:
            net = fullyconnected(net, W_shape=[shape, fc_n_hidden], b_shape=[fc_n_hidden], 
                                 regularizer=regularizer, weight_decay=weight_decay, 
                                 name='casCNN_FC')
    return net

def CasRes(x, is_training, conv_filter_shape, resnet_filter_shape, n_blocks, downsample, 
           regularizer='L2', weight_decay=0.0001, fc_layer=True, fc_n_hidden=1024, bias=True,
           reuse=True, **kwargs):
    # Vanilla-CNN Module
    if reuse:
        net = ConvNet(x, filter_shape=conv_filter_shape, 
                      regularizer='L2', weight_decay=weight_decay, 
                      scope='Conv2D',
                      reuse=tf.AUTO_REUSE)
    else:
        net = ConvNet(x, filter_shape=conv_filter_shape, 
                      regularizer='L2', weight_decay=weight_decay)
        
    net = tf.nn.relu(net)
    
    # ResNet Module
    downsample_stride = 2    

    if reuse:
        net = ResNet(is_training, net, n_blocks,
                     downsample=downsample,
                     filter_shape=resnet_filter_shape,
                     downsample_stride=downsample_stride,
                     regularizer=regularizer, 
                     weight_decay=weight_decay,
                     reuse=tf.AUTO_REUSE)
    else:
        net = ResNet(is_training, net, n_blocks,
                     downsample=downsample,
                     filter_shape=resnet_filter_shape,
                     downsample_stride=downsample_stride,
                     regularizer=regularizer, 
                     weight_decay=weight_decay)
        
#     # Average Pooling
#     ksize = net.get_shape().as_list()[1:-1]
#     strides = ksize = [1] + ksize + [1]
#     net = tf.nn.avg_pool(net, ksize=ksize, strides=strides, padding='SAME')
#     net = tf.reshape(net, [-1, net.get_shape().as_list()[-1]])

    shape_list = net.get_shape().as_list()[1:]
    shape = 1
    for i in shape_list:
        shape *= i
    net = tf.reshape(net, [-1, shape])
    
    if fc_layer:
        if reuse:
            net = fullyconnected(net, W_shape=[shape, fc_n_hidden], b_shape=[fc_n_hidden], 
                                 regularizer=regularizer, weight_decay=weight_decay, 
                                 scope='casCNN_FC',
                                 reuse=tf.AUTO_REUSE)
        else:
            net = fullyconnected(net, W_shape=[shape, fc_n_hidden], b_shape=[fc_n_hidden], 
                                 regularizer=regularizer, weight_decay=weight_decay, 
                                 name='casCNN_FC')
        net = tf.nn.relu(net)
    return net

def CasRNN(net, n_hidden=[64], n_classes=5,
           bidirect=False, regularizer='L2', weight_decay=0.0001, **kwargs):
    """
    Arguments;
        n_hidden: list
    """
    for i, n in enumerate(n_hidden):
        if bidirect:
            # Forward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n, forget_bias=1.0, name='LSTM_fw_{}'.format(i))
            # Backward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n, forget_bias=1.0, name='LSTM_bw_{}'.format(i))
            # Get BiRNN cell output
            net, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n, name='LSTM_{}'.format(i))
            net, _ = tf.contrib.rnn.static_rnn(lstm_cell, net, dtype=tf.float32)
        
    W_0 = n_hidden[-1]
    if bidirect:
        W_0 = W_0 * 2
    net = fullyconnected(net[-1], 
                         W_shape=[W_0, n_classes], b_shape=[n_classes], 
                         regularizer=regularizer, weight_decay=weight_decay, 
                         scope='casRNN_FC')
    return net