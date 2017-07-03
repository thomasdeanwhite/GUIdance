# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import random
import csv
import os
from six.moves import cPickle as pickle

input_length = 1028
output_length = 4


x = tf.placeholder(tf.float32, [None, input_length])
y = tf.placeholder(tf.float32, [None, output_length])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([input_length, 1024], stddev=0.05), name='W1')
b1 = tf.Variable(tf.random_normal([1024]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([1024, 128], stddev=0.05), name='W2')
b2 = tf.Variable(tf.random_normal([128]), name='b2')

# and the weights connecting the hidden layer to the output layer
W3 = tf.Variable(tf.random_normal([128, 64], stddev=0.05), name='W3')
b3 = tf.Variable(tf.random_normal([64]), name='b3')

# and the weights connecting the hidden layer to the output layer
W4 = tf.Variable(tf.random_normal([64, output_length], stddev=0.05), name='W4')
b4 = tf.Variable(tf.random_normal([output_length]), name='b4')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
#hidden_out = tf.nn.tanh(hidden_out)

# calculate the output of the hidden layer
hidden_out2 = tf.add(tf.matmul(hidden_out, W2), b2)
#hidden_out2 = tf.nn.tanh(hidden_out2)

# calculate the output of the hidden layer
hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)
#hidden_out3 = tf.nn.tanh(hidden_out3)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
#y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out3, W4), b4))
y_ = tf.add(tf.matmul(hidden_out3, W4), b4)

# now let's define the cost function which we are going to train the model on
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                                               + (1 - y) * tf.log(1 - y_clipped), axis=1))

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

def get_output(sess, data_row):
    return sess.run(y_, feed_dict={x:np.array([data_row])})

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    # Restore variables from disk.
    model_file = "NuiMimic/default-user/model/model.ckpt"
    if os.path.isfile("NuiMimic/default-user/model/checkpoint"):
        saver.restore(sess, model_file)

    input = []

    for i in range(1, len(sys.argv)):
        input.append(float(sys.argv[i]))

    input = np.array(input)

    #model is loaded! Use model:
    r = get_output(sess, input)

    print(r)


