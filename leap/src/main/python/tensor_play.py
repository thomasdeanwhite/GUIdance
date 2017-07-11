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


hidden_layers = [1024, 256, 64]

x = tf.placeholder(tf.float32, [None, input_length])
y = tf.placeholder(tf.float32, [None, output_length])

def generate_layer_defaults(dims, min=-0.1, max=0.1):
    return tf.random_uniform(dims, minval=min, maxval=max)

#Create 3 hidden layers
W1 = tf.Variable(generate_layer_defaults([input_length, hidden_layers[0]]), name='W1')
b1 = tf.Variable(generate_layer_defaults([hidden_layers[0]]), name='b1')

W2 = tf.Variable(generate_layer_defaults([hidden_layers[0], hidden_layers[1]]), name='W2')
b2 = tf.Variable(generate_layer_defaults([hidden_layers[1]]), name='b2')

W3 = tf.Variable(generate_layer_defaults([hidden_layers[1], hidden_layers[2]]), name='W3')
b3 = tf.Variable(generate_layer_defaults([hidden_layers[2]]), name='b3')

W4 = tf.Variable(generate_layer_defaults([hidden_layers[2], output_length]), name='W4')
b4 = tf.Variable(generate_layer_defaults([output_length],), name='b4')

hidden_out = tf.add(tf.matmul(x, W1), b1)

hidden_out2 = tf.add(tf.matmul(hidden_out, W2), b2)

hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)

y_ = tf.add(tf.matmul(hidden_out3, W4), b4)

# now let's define the cost function which we are going to train the model on
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                                               + (1 - y) * tf.log(1 - y_clipped), axis=1))

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

def get_output(sess, data_row):
    return sess.run(y_, feed_dict={x:data_row})

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    # Restore variables from disk.
    model_file = "NuiMimic/default-user/model/model.ckpt"
    if os.path.isfile("NuiMimic/default-user/model/checkpoint"):
        saver.restore(sess, model_file)

    inpt = []

    while true:

        app_inpt = input()

        for i in range(len(app_inpt)):
            inpt.append(float(app_inpt[i]))

        inpt = np.array([inpt])

        #model is loaded! Use model:
        r = get_output(sess, inpt)

        print(r[0][0], r[0][1], r[0][2], r[0][3])


