# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import csv
import os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import math


learning_rate = 0.0001
epochs = 1000
batch_size = 5
percent_training = 0.7
percent_testing = 1
percent_validation = 0.5
hidden_layers = [4900, 4096, 256]
sd = 0.005
sdw = 0.005
show_output_image = True
image_height = 64

data = []
output = []

with open('training_inputs.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    counter = 0
    for row in reader:
        data.append([])
        for e in row:
            data[counter].append(float(e))
        counter += 1

with open('training_outputs.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    counter = 0
    for row in reader:
        output.append([])
        for e in row:
            output[counter].append(float(e))
        counter += 1

data = np.array(data)
data = data[:, 0:image_height*image_height]
output = np.array(output)

print("Data Shape:", data.shape)
print("Output Shape:", output.shape)


def get_sample(data, output, n):
    samples = random.sample(range(data.shape[0]), n)
    return data[samples], output[samples], samples

train_dataset, train_labels, samples = get_sample(data, output, int(percent_training * data.shape[0]))
data = np.delete(data, samples, axis=0)

valid_dataset, valid_labels, samples = get_sample(data, output, int(percent_validation * data.shape[0]))
data = np.delete(data, samples, axis=0)

test_dataset, test_labels, _ = get_sample(data, output, int(percent_testing * data.shape[0]))


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

x = tf.placeholder(tf.float32, [None, train_dataset.shape[1]])
y = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

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

print(W_conv1.shape)

x_image = tf.reshape(x, [-1, image_height, image_height, 1])

h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_pool1 = max_pool_2x2(h_conv1)

print(h_conv1.shape)


W_conv2 = weight_variable([8, 8, 4, 8])
b_conv2 = bias_variable([8])

print(W_conv2.shape)

h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = max_pool_2x2(h_conv2)

print(h_conv2.shape)


W_fc1 = weight_variable([16*16*8, 256])
b_fc1 = bias_variable([256])

print(W_fc1.shape)

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*8])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

print(h_pool2_flat.shape)

keep_prob = 1.0
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 4])
b_fc2 = bias_variable([4])

print(W_fc2.shape)

y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

print(y_.shape)

print("model created successfully")

loss = tf.losses.huber_loss(y, y_)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(y, y_)), 6.0))

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf')

saver = tf.train.Saver()

plt.close('all')

plt.figure(1)

plots = 1

def getActivations(sess, layer,stimuli, count, plot_orig):
    units = sess.run(layer,feed_dict={x:[stimuli]})
    plotNNFilter(units, stimuli, count, plot_orig)

def plotNNFilter(units, stimuli, count, plot_orig):
    filters = units.shape[3]
    n_columns = 13
    n_rows = 5
    if plot_orig:
        plt.subplot(n_rows, n_columns, count + 1)
        plt.title('Inp')
        plt.imshow(np.reshape(stimuli, [64, 64]), cmap="gray")
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+2+count)
        plt.title('F ' + str(i))
        plt.imshow(units[0,:,:,i], cmap="gray")

# start the session
with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())


    # Restore variables from disk.
    model_file = "model/model.ckpt"
    # if os.path.isfile("model/checkpoint"):
    #     saver.restore(sess, model_file)
    #     print("Model restored.")

    total_len = train_labels.shape[0]

    count = 0

    for i in range(data.shape[0]):
        plt.figure(1, figsize=(20,20))

        getActivations(sess, h_pool1, data[i], count, True)
        count += 4

        getActivations(sess, h_pool2, data[i], count, False)
        count += 9

        if count > 64:
            plt.show()
            count = 0

