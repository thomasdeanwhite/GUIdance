# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt

input_length = 4100
output_length = 4

wd = os.getcwd()

image_height = 64
image_features = image_height * image_height

rem_features = 4

hidden_layers = [1024, 256, 64]

x = tf.placeholder(tf.float32, [None, image_features + rem_features])

x_img_raw = tf.slice(x, [0, 0], [-1, image_features])

x_img = tf.subtract(tf.multiply(x_img_raw, 2.0), 1.0)

x_rem = tf.slice(x, [0, image_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, output_length])

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weight_variable([8, 8, 1, 4])
b_conv1 = bias_variable([4])

x_image = tf.reshape(x_img, [-1, image_height, image_height, 1])

h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))


W_conv2 = weight_variable([8, 8, 4, 8])
b_conv2 = bias_variable([8])

h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))


W_fc1 = weight_variable([16*16*8, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*8])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

h_fcl_joined = tf.concat([h_fc1, x_rem], 1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fcl_joined, keep_prob)

W_fc2 = weight_variable([1024+rem_features, 256])
b_fc2 = bias_variable([256])

h_fcl2 = tf.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([256, rem_features])
b_fc3 = bias_variable([rem_features])

y_ = tf.tanh(tf.matmul(h_fcl2, W_fc3) + b_fc3)

# now let's define the cost function which we are going to train the model on
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                                               + (1 - y) * tf.log(1 - y_clipped), axis=1))

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

def get_output(sess, data_row):
    return sess.run(y_, feed_dict={x:data_row, keep_prob:1.0})

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    # Restore variables from disk.
    model_file = wd + "/model/model.ckpt"
    if os.path.isfile(wd + "/model/checkpoint"):
        saver.restore(sess, model_file)
        print("Model restored.")



    while True:

        inpt = []
        user_inpt = input()

        app_inpt = user_inpt.split(" ")

        for i in range(len(app_inpt)):
            inpt.append(float(app_inpt[i]))

        inpt = np.array([inpt])

        #model is loaded! Use model:
        r = get_output(sess, inpt)

        print("dl-result", r[0][0], r[0][1], r[0][2], r[0][3])



