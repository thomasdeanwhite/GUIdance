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
from autoencoder import AutoEncoder
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
image_height = 64

raw_data = []
output = []
wd = os.getcwd()

print(wd)

file_Name = wd + "/data.pickle"

if os.path.isfile(file_Name):
    print("Restoring pickled data")
    fileObject = open(file_Name,'rb')
    raw_data = pickle.load(fileObject)

    fileObject = open(wd + "/output.pickle",'rb')
    output = pickle.load(fileObject)
else:
    print("Pickling data")
    for i in range(1, len(sys.argv)):
        os.chdir(os.path.join(wd, sys.argv[i]))
        print("loading", sys.argv[i])
        with open('training_inputs.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            counter = len(raw_data)
            for row in reader:
                raw_data.append([])
                for e in row:
                    raw_data[counter].append(float(e))
                counter += 1

        with open('training_outputs.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            counter = len(output)
            for row in reader:
                output.append([])
                for e in row:
                    output[counter].append(float(e))
                counter += 1
    raw_data = np.array(raw_data)
    output = np.array(output)
    fileObject = open(file_Name,'wb')
    pickle.dump(raw_data,fileObject)
    fileObject.close()
    fileObject = open(wd + "/output.pickle",'wb')
    pickle.dump(output, fileObject)
    fileObject.close()

def get_sample(data, output, n):
    random.seed(1)
    samples = random.sample(range(data.shape[0]), n)
    return data[samples], output[samples], samples

def shape_to_string(frame):
    return ("(" + str(frame.shape[0]) + ", " + str(frame.shape[1]) + ")")


compressed_features = 64*64

#pca = decomposition.PCA(n_components=compressed_features, whiten=True)
fileObject = open(wd + "/pca.pickle",'rb')
preop = pickle.load(fileObject)

def whiten_data(d):
    global preop
    d = preop['pca'].transform(d)
    #d = preop['scalar'].transform(d)
    return d

image_features = image_height * image_height

#raw_data = raw_data[:, :]

whitened_data = np.copy(raw_data[:, 0:image_features])
# print("Fitting data")
# pca.fit(whitened_data)

print("Whitening data")

# preop['pca'] = decomposition.PCA(n_components=compressed_features, whiten=True)
#
# preop['pca'].fit(whitened_data)

whitened_data = whiten_data(whitened_data)

# scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#
# preop['scalar'] = scalar

preop['scalar'].fit(whitened_data)

whitened_data = preop['scalar'].transform(whitened_data)

print("Whitened Data: mean: " + str(np.mean(whitened_data)) + " var: " + str(np.var(whitened_data)) +
      " range: (" + str(np.min(whitened_data)) + "," + str(np.max(whitened_data)) + ")")

#whitened_data = preop['scalar'].transform(whitened_data)

data = np.copy(raw_data[:, image_features:(image_features+4)])


data = np.insert(data, [0], whitened_data, axis=1)

print(shape_to_string(data))

raw_data, output, samp = get_sample(raw_data, output, 30)

data = data[samp]

whitened_data = data[:, 0:compressed_features]

rem_features = data.shape[1] % compressed_features

#data = data[:, 0:image_height*image_height]
output = np.array(output)

print("Data Shape:", data.shape)
print("Output Shape:", output.shape)

x = tf.placeholder(tf.float32, [None, data.shape[1]])

x_img_raw = tf.slice(x, [0, 0], [-1, compressed_features])

x_rem = tf.slice(x, [0, compressed_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, output.shape[1]])

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1.5, maxval=1.5)
    return tf.Variable(initial)

# W_img_inpt = weight_variable([compressed_features, 24*24])
# b_img_inpt = bias_variable([24*24])
#
# W_img_inpt2 = weight_variable([24*24, 16*16])
# b_img_inpt2 = bias_variable([16*16])
#
# h_auto_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(x_img_raw, W_img_inpt), b_img_inpt))
#
# h_auto_fc2 = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc1, W_img_inpt2), b_img_inpt2))
#
# W_auto_decoder = weight_variable([16*16, 24*24])
# b_auto_decoder = bias_variable([24*24])
#
# W_auto_decoder2 = weight_variable([24*24, compressed_features])
# b_auto_decoder2 = bias_variable([compressed_features])
#
# auto_encoder_step = tf.nn.sigmoid(tf.add(tf.matmul(h_auto_fc2, W_auto_decoder), b_auto_decoder))
#
# auto_encoder_ = tf.nn.sigmoid(tf.add(tf.matmul(auto_encoder_step, W_auto_decoder2), b_auto_decoder2))
#
# loss_auto_encoder = tf.reduce_mean(tf.pow(x_img_raw - auto_encoder_, 2))

auto_encoder = AutoEncoder(x_img_raw)

train_auto_encoder_step = tf.train.RMSPropOptimizer(learning_rate).minimize(auto_encoder.loss)

accuracy_auto_encoder = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(x_img_raw, auto_encoder.output_layer)), 2.0))

auto_encoder_out = tf.multiply(tf.add(auto_encoder.output_layer, 1.0), 0.5)

h_fcl_joined = tf.concat([auto_encoder.output_layer, x_rem], 1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fcl_joined, keep_prob)

W_fc2 = weight_variable([compressed_features+rem_features, 256])
b_fc2 = bias_variable([256])

h_fcl2 = tf.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([256, rem_features])
b_fc3 = bias_variable([rem_features])

y_ = tf.tanh(tf.matmul(h_fcl2, W_fc3) + b_fc3)

saver = tf.train.Saver()

plt.close('all')

plt.figure(1)

plots = 1

image_size = [10, 13]

# start the session
def run():
    global data
    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())

        figs = 30


        # Restore variables from disk.
        model_file = wd + "/model_encoding/model.ckpt"
        if os.path.isfile(wd + "/model_encoding/checkpoint"):
            saver.restore(sess, model_file)
            print("Model restored.")
        count = 0

        #random.shuffle(data)

        stimuli = raw_data[:, 0:image_features]
        stimuli_whitened = preop['pca'].inverse_transform(preop['scalar'].inverse_transform(data[:, 0:compressed_features]))

        print("")

        #data_whitened, U = whiten_data(stimuli)

        results = preop['pca'].inverse_transform(preop['scalar'].inverse_transform(sess.run(auto_encoder.output_layer, feed_dict={x:data})))
        print("Acc: " + str(sess.run(accuracy_auto_encoder, feed_dict={x: data})))
        print("Loss: " + str(sess.run(auto_encoder.loss, feed_dict={x: data})))

        total_rows = math.ceil(figs / 5)

        while count < (figs/3):

            # plot_num = count % 5
            # plot_row = math.floor(count / 5)

            plt.subplot(5, total_rows, count*3 + 1)
            #plt.title('Inp')
            plt.imshow(np.reshape(stimuli[count,:], [image_height, image_height]), cmap="gray")


            plt.subplot(5, total_rows, count*3 + 2)
            #plt.title('Inp')
            plt.imshow(np.reshape(stimuli_whitened[count,:], [image_height, image_height]), cmap="gray")


            plt.subplot(5, total_rows, count*3 + 3)
            #plt.title('Inp')
            plt.imshow(np.reshape(results[count, :], [image_height, image_height]), cmap="gray")
            count = count + 1


        plt.show()

run()