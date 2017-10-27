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
import sys
import random
from sklearn.decomposition import PCA


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
image_height = 64

data = []
output = []
wd = os.getcwd()
for i in range(1, len(sys.argv)):
    os.chdir(os.path.join(wd, sys.argv[i]))
    print("loading", sys.argv[i])
    with open('training_inputs.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = len(data)
        for row in reader:
            data.append([])
            for e in row:
                data[counter].append(float(e))
            counter += 1

    with open('training_outputs.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = len(output)
        for row in reader:
            output.append([])
            for e in row:
                output[counter].append(float(e))
            counter += 1

data = np.array(data)

def whiten_data(d):

    new_d = d - d.mean(axis=0)

    new_d = new_d / np.sqrt((new_d ** 2).sum(axis=1))[:,None]

    cov = np.cov(new_d, rowvar=True)

    U, S, V = np.linalg.svd(cov)

    z_mat = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + 1E-5)), U.T))

    new_d = np.dot(z_mat, new_d)

    return new_d, U


print("Whitening data")
image_features = image_height * image_height

whitened_data, U = whiten_data(data[:, 0:image_features])

np.copyto(data, whitened_data, where)

rem_features = data.shape[1] % image_features

print(data.shape)

#data = data[:, 0:image_height*image_height]
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

x_img_raw = tf.slice(x, [0, 0], [-1, image_features])

x_img = tf.subtract(tf.multiply(x_img_raw, 2.0), 1.0)

x_rem = tf.slice(x, [0, image_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# this layer will compress screenshot by factor of 4 in both dimensions
# across two layers
W_img_inpt = weight_variable([image_features, 32*32])
b_img_inpt = bias_variable([32*32])

W_img_inpt2 = weight_variable([32*32, 16*16])
b_img_inpt2 = bias_variable([16*16])

h_auto_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(x_img_raw, W_img_inpt), b_img_inpt))

h_auto_fc2 = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc1, W_img_inpt2), b_img_inpt2))

W_auto_decoder = weight_variable([16*16, 32*32])
b_auto_decoder = bias_variable([32*32])

W_auto_decoder2 = weight_variable([32*32, image_features])
b_auto_decoder2 = bias_variable([image_features])

auto_encoder_step = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc2, W_auto_decoder), b_auto_decoder))

auto_encoder_ = tf.nn.sigmoid(tf.add(tf.matmul(auto_encoder_step, W_auto_decoder2), b_auto_decoder2))

loss_auto_encoder = tf.reduce_mean(tf.pow(x_img_raw - auto_encoder_, 2))

train_auto_encoder_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_auto_encoder)

accuracy_auto_encoder = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(x_img_raw, auto_encoder_)), 2.0))

auto_encoder_out = tf.multiply(tf.add(auto_encoder_, 1.0), 0.5)

h_fcl_joined = tf.concat([auto_encoder_, x_rem], 1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fcl_joined, keep_prob)

W_fc2 = weight_variable([image_features+rem_features, 256])
b_fc2 = bias_variable([256])

h_fcl2 = tf.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([256, rem_features])
b_fc3 = bias_variable([rem_features])

y_ = tf.tanh(tf.matmul(h_fcl2, W_fc3) + b_fc3)

print(y_.shape)

saver = tf.train.Saver()

plt.close('all')

plt.figure(1)

plots = 1

image_size = [10, 13]

def get_image(sess, ds, width, height, fn):
    res = sess.run(fn, feed_dict={x:[ds]})
    #res = tf.reduce_sum(tf.transpose(res), keep_dims=True)
    return res

def getActivations(sess, layer,stimuli, count, plot_orig):
    units = sess.run(layer,feed_dict={x:[stimuli]})
    plotNNFilter(units, stimuli, count, plot_orig)

def plotNNFilter(units, stimuli, count, plot_orig):
    global image_size
    filters = units.shape[3]
    n_columns = image_size[1]
    n_rows = image_size[0]

    stimuli = stimuli[0:image_features]

    if plot_orig:
        plt.subplot(n_rows, n_columns, count + 1)
        #plt.title('Inp')
        plt.imshow(np.reshape(stimuli, [image_height, image_height]), cmap="gray")
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+2+count)
        #plt.title('F ' + str(i))
        plt.imshow(units[0,:,:,i], cmap="gray")

def show_image(ds, width, height):
    global plots
    ds = ds[0:width*height]
    ds = np.reshape(ds, [width, height])
    plt.subplot(5, 2, plots)
    plots += 1
    plt.imshow(ds, cmap="gray")

# start the session
def run():
    global data
    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())

        figs = 30


        # Restore variables from disk.
        model_file = wd + "/model_encoding/model.ckpt"
        # if os.path.isfile(wd + "/model_encoding/checkpoint"):
        #     saver.restore(sess, model_file)
        #     print("Model restored.")
        count = 0

        #random.shuffle(data)

        stimuli = data[:, 0:image_features]

        data_whitened, U = whiten_data(stimuli)

        results = sess.run(auto_encoder_, feed_dict={x:data})
        print("Acc: " + str(sess.run(accuracy_auto_encoder, feed_dict={x: data})))
        print("Loss: " + str(sess.run(loss_auto_encoder, feed_dict={x: data})))

        total_rows = math.ceil(figs / 5)

        while count < (figs/3):

            # plot_num = count % 5
            # plot_row = math.floor(count / 5)

            plt.subplot(5, total_rows, count*3 + 1)
            #plt.title('Inp')
            plt.imshow(np.reshape(stimuli[count,:], [image_height, image_height]), cmap="gray")


            plt.subplot(5, total_rows, count*3 + 2)
            #plt.title('Inp')
            plt.imshow(np.reshape(data_whitened[count,:], [image_height, image_height]), cmap="gray")


            plt.subplot(5, total_rows, count*3 + 3)
            #plt.title('Inp')
            plt.imshow(np.reshape(results[count,:], [image_height, image_height]), cmap="gray")
            count = count + 1


        plt.show()

run()