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
import matplotlib.image as mpimg
from autoencoder import AutoEncoder

learning_rate = 0.0001
balance_data = True
epochs = 1000
batch_size = 25
percent_training = 0.7
percent_testing = 1
percent_validation = 0.5
hidden_layers = [4900, 4096, 256]
sd = 0.005
sdw = 0.005
show_output_image = True
image_height = 64
compressed_features = 64*64
rem_features = 8
raw_data = []
output = []
wd = os.getcwd()

print(wd)

file_Name = wd + "/data.pickle"

image_features = image_height * image_height

def to_gray(rgb):
    return np.dot(rgb[...,:3], [0.3333, 0.3333, 0.3333])

img = to_gray(mpimg.imread('screenshot.png'))

print(img.shape)

segments = []

images_y = int(img.shape[0])
images_x = int(img.shape[1])

print("Converting screenshot to", images_x, "x", images_y, "sub-images (", images_x*images_y, "total)")
preop = {}
with open('preop.pickle','rb') as f:
    preop = pickle.load(f)

def inverse_proep(data):
    global preop
    data = data * preop['std']
    data = data + preop['mean']
    return data

x = tf.placeholder(tf.float32, [None, image_features])

x_img_raw = tf.slice(x, [0, 0], [-1, compressed_features])

x_rem = tf.slice(x, [0, compressed_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, rem_features])

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

auto_encoder = AutoEncoder(x_img_raw)

loss_auto_encoder = tf.reduce_mean(tf.square(x_img_raw - auto_encoder.output_layer))

learning_rate = tf.placeholder(tf.float32)

train_auto_encoder_step = tf.train.AdadeltaOptimizer(learning_rate, 0.95, 1e-08, False).minimize(loss_auto_encoder)

global_step = tf.placeholder(tf.int64)

train_auto_encoder_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_auto_encoder)

accuracy_auto_encoder = tf.add(1.0,
                               -tf.div(tf.reduce_mean(tf.losses.absolute_difference(x_img_raw, auto_encoder.output_layer)), 2.0))


def leaky_relu(x, alpha):
    return tf.maximum(alpha*x,x)


#Start of the DNN to calculated screenshot interaction type
keep_prob = tf.placeholder(tf.float32)

x_inp = tf.placeholder(tf.float32, [None, auto_encoder.minimal_layer_size])

# x_mean_features, x_standard_deviation = tf.nn.moments(x_inp, axes=[0])
#
# x_standardised = (x_inp - x_mean_features) / x_standard_deviation

W_fc2 = weight_variable([auto_encoder.minimal_layer_size, 512])
b_fc2 = bias_variable([512])

h_fcl2 = leaky_relu(tf.matmul(x_inp, W_fc2) + b_fc2, 0.1)#, keep_prob)
# h_fcl2_conv = tf.layers.conv2d(x_inp, 7, 64, activation=tf.nn.relu)
# h_fcl2 = tf.layers.max_pooling2d(h_fcl2_conv, 2, 2)

W_fc3 = weight_variable([512, rem_features])
b_fc3 = bias_variable([rem_features])

h_fcl3 = tf.nn.tanh(tf.nn.dropout(tf.matmul(h_fcl2, W_fc3) + b_fc3, keep_prob))

# layers = []
# vars = []
#
# for i in range(rem_features):
#     W_fc2 = weight_variable([auto_encoder.minimal_layer_size, 512])
#     b_fc2 = bias_variable([512])
#
#     vars.append(W_fc2)
#     vars.append(b_fc2)
#
#     h_fcl2 = tf.nn.relu(tf.matmul(x_inp, W_fc2) + b_fc2)#, keep_prob)
#
#     W_fc3 = weight_variable([512, 1])
#     b_fc3 = bias_variable([1])
#
#     vars.append(W_fc3)
#     vars.append(b_fc3)
#
#     h_fcl3 = tf.nn.sigmoid(tf.nn.dropout(tf.matmul(h_fcl2, W_fc3) + b_fc3, keep_prob))
#
#     layers.append(h_fcl3)
#
#
# #
# # W_fc4 = weight_variable([32, rem_features])
# # b_fc4 = bias_variable([rem_features])
# concat_layer = layers[0]
# for i in range(1, len(layers)):
#     concat_layer = tf.concat([concat_layer, layers[i]], axis=1)

#y_ = concat_layer

y_ = tf.nn.sigmoid(tf.matmul(h_fcl2, W_fc3) + b_fc3)

out = tf.nn.softmax(y_)

#y_ = tf.layers.dense(h_fcl2, rem_features)
print("Output layer", y_.shape)
print("model created successfully")

#loss = tf.reduce_mean(tf.square(tf.subtract(y, y_)), axis=0)
loss = tf.losses.softmax_cross_entropy(y, y_)
#epsilon = 0.00001
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.add(y_, epsilon)), reduction_indices=[1]))

# learning_rate = tf.placeholder(tf.float32, shape=[])

#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate, epsilon=0.001).minimize(loss)

accuracy = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(y, y_)), 2.0))


saver_auto = tf.train.Saver(auto_encoder.variables)

#saver = tf.train.Saver([W_fc2, W_fc3, b_fc2, b_fc3, W_fc4, b_fc4])
saver = tf.train.Saver([W_fc2, W_fc3, b_fc2, b_fc3])
plt.close('all')

plt.figure(1)

plots = 1

image_size = [10, 13]

with tf.Session() as sess:

    with open("screenshot_out.csv", "w") as myfile:
        myfile.write("x,y,var,val\n")

    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    model_file = wd + "/model_encoding/model.ckpt"
    if os.path.isfile(wd + "/model_encoding/checkpoint"):
        saver_auto.restore(sess, model_file)
        print("Autoencoder restored.")

    model_file = "model/model.ckpt"
    if os.path.isfile("model/checkpoint"):
        saver.restore(sess, model_file)
        print("Model restored.")

    # start the session
    def run(i):
        global data, x, x_inp, images_x, images_y, image_height


        encoder_results = sess.run(auto_encoder.minimal_layer, feed_dict={x:data, auto_encoder.keep_prob: 1.0})

        results = sess.run(y_, feed_dict={x_inp: encoder_results, keep_prob: 1.0})

        for j in range(images_x):
            xn = i * image_height + (image_height/2)
            yn = int(j * image_height + (image_height/2))

            xn = max(xn, 0)
            xn = min(xn, images_y - image_height)

            yn = max(yn, 0)
            yn = min(yn, images_x - image_height)

            index = yn#(j * images_x)

            with open("screenshot_out.csv", "a") as myfile:
                myfile.write(str(xn) + "," + str(yn) + "," + "LeftClick," + str(results[index, 0]) + "\n")


    for i in range(images_y):
        i = max(i, 0)
        i = min(i, images_y - image_height)
        print(i, "of", images_y)
        segments = []
        for j in range(images_x):
            j = max(j, 0)
            j = min(j, images_x - image_height)
            segment_row = img[i:i+image_height, j:j+image_height]
            segments.append(np.reshape(segment_row, [image_features]))

        segments = np.array(segments)

        raw_data = segments

        def get_sample(data, output, n):
            random.seed(1)
            samples = random.sample(range(data.shape[0]), n)
            return data[samples], output[samples], samples

        def shape_to_string(frame):
            return ("(" + str(frame.shape[0]) + ", " + str(frame.shape[1]) + ")")


        whitened_data = raw_data



        whitened_data = whitened_data - preop['mean']

        whitened_data = whitened_data / preop['std']


        data = whitened_data

        # data = np.copy(raw_data[:, image_features:(image_features+rem_features)])
        #
        #
        # data = np.insert(data, [0], whitened_data, axis=1)
        #
        # whitened_data = data[:, 0:compressed_features]

        #data = data[:, 0:image_height*image_height]
        output = np.array(output)

        run(i)



