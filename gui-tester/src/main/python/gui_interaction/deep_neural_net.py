# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import csv
import os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import math
import sys
import random
from sklearn import decomposition
from sklearn import preprocessing

learning_rate = 0.0001
epochs = 1000
batch_size = 25
percent_training = 0.7
percent_testing = 1
percent_validation = 0.5
hidden_layers = [4900, 4096, 256]
sd = 0.005
sdw = 0.005
show_output_image = True
image_height = 28
compressed_features = 28*28
rem_features = 4


x = tf.placeholder(tf.float32, [None, compressed_features+4])

x_img_raw = tf.slice(x, [0, 0], [-1, compressed_features])

x_rem = tf.slice(x, [0, compressed_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, 4])

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=1)
    return tf.Variable(initial)

W_img_inpt = weight_variable([compressed_features, 24*24])
b_img_inpt = bias_variable([24*24])

W_img_inpt2 = weight_variable([24*24, 16*16])
b_img_inpt2 = bias_variable([16*16])

h_auto_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(x_img_raw, W_img_inpt), b_img_inpt))

h_auto_fc2 = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc1, W_img_inpt2), b_img_inpt2))

W_auto_decoder = weight_variable([16*16, 24*24])
b_auto_decoder = bias_variable([24*24])

W_auto_decoder2 = weight_variable([24*24, compressed_features])
b_auto_decoder2 = bias_variable([compressed_features])

auto_encoder_step = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc2, W_auto_decoder), b_auto_decoder))

auto_encoder_ = tf.nn.sigmoid(tf.add(tf.matmul(auto_encoder_step, W_auto_decoder2), b_auto_decoder2))

loss_auto_encoder = tf.reduce_mean(tf.pow(x_img_raw - auto_encoder_, 2))

train_auto_encoder_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_auto_encoder)

accuracy_auto_encoder = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(x_img_raw, auto_encoder_)), 2.0))

auto_encoder_out = tf.multiply(tf.add(auto_encoder_, 1.0), 0.5)

h_fcl_joined = tf.concat([auto_encoder_, x_rem], 1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fcl_joined, keep_prob)

W_fc2 = weight_variable([compressed_features+rem_features, 256])
b_fc2 = bias_variable([256])

h_fcl2 = tf.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([256, rem_features])
b_fc3 = bias_variable([rem_features])

y_ = tf.tanh(tf.matmul(h_fcl2, W_fc3) + b_fc3)