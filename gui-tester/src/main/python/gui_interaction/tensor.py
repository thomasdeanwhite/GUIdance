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
from sklearn import decomposition
from sklearn import preprocessing
from autoencoder import AutoEncoder


learning_rate_start = 1.0
learning_rate_min = 0.005
learning_rate_decay = 0.9995
epochs = 5000000
batch_size = 10000
percent_training = 0.9
percent_testing = 1
percent_validation = 0.9
image_height = 64
TRAIN_MODEL = True
TRAIN_AUTOENCODER = False
compressed_features = 64*64
raw_data = []
output = []
balance_data = True
wd = os.getcwd()
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
os.chdir(wd)
raw_data = np.array(raw_data)


#raw_data = np.insert(raw_data, [compressed_features], output, axis=1)

output = np.array(output)

image_features = image_height * image_height

print("Raw data shape:", raw_data.shape)

whitened_data = np.copy(raw_data[:, 0:image_features]).astype(float)
print("Fitting data")
#whitened_data = whitened_data / 255


mean_features = np.mean(whitened_data, axis=0)
standard_deviation = np.std(whitened_data, axis=0)

preop = {
    'mean':mean_features,
    'std':standard_deviation
}

with open('preop.pickle', 'wb') as f:
    pickle.dump(preop, f)

whitened_data = whitened_data - mean_features

whitened_data = whitened_data / standard_deviation

rem_features = output.shape[1]

print("Image Data: mean: " + str(np.mean(whitened_data)) + " var: " + str(np.var(whitened_data)) +
      " range: (" + str(np.min(whitened_data)) + "," + str(np.max(whitened_data)) + ")")
data = np.copy(raw_data[:, image_features:(image_features + rem_features)])

data = np.insert(data, [0], whitened_data, axis=1)



output = np.array(output)

print("Data Shape:", data.shape, output.shape)


def get_sample(data, output, n):
    samples = random.sample(range(data.shape[0]), n)
    return data[samples], output[samples], samples


print("Writing proc data")
# np.savetxt("data.csv", whitened_data[0:2000, :].flatten(), delimiter=",")

train_dataset, train_labels, samples = get_sample(data, output, int(percent_training * data.shape[0]))
data = np.delete(data, samples, axis=0)

valid_dataset, valid_labels, samples = get_sample(data, output, int(percent_validation * data.shape[0]))
data = np.delete(data, samples, axis=0)

test_dataset, test_labels, _ = get_sample(data, output, int(percent_testing * data.shape[0]))

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

x = tf.placeholder(tf.float32, [None, data.shape[1]])

x_img_raw = tf.slice(x, [0, 0], [-1, compressed_features])

# x_img = tf.subtract(tf.multiply(x_img_raw, 2.0), 1.0)

x_rem = tf.slice(x, [0, compressed_features], [-1, -1])

y = tf.placeholder(tf.float32, [None, rem_features])


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.5)
    return tf.Variable(initial)


auto_encoder = AutoEncoder(x_img_raw)



loss_auto_encoder = auto_encoder.loss



learning_rate = tf.placeholder(tf.float32)

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

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf')



plt.close('all')

plt.figure(1)

plots = 1
#saver = tf.train.Saver([W_fc2, W_fc3, b_fc2, b_fc3])
saver = tf.train.Saver([W_fc2, W_fc3, b_fc2, b_fc3])
saver_auto = tf.train.Saver(auto_encoder.variables)
# start the session
if TRAIN_MODEL:
    with tf.Session() as sess:
        # learning_rate_start = 1.0
        # learning_rate_min = 0.0001
        sess.run(tf.global_variables_initializer())

        print("Filtered training data:", train_dataset.shape)

        model_file = wd + "/model_encoding/model.ckpt"
        if os.path.isfile(wd + "/model_encoding/checkpoint"):
            saver_auto.restore(sess, model_file)
            print("Auto-encoder restored.")

        # Restore variables from disk.
        model_file = "model/model.ckpt"
        # if os.path.isfile("model/checkpoint"):
        #     saver.restore(sess, model_file)
        #     print("Model restored.")

        total_len = train_labels.shape[0]

        count = 0

        learn_rate = learning_rate_start

        encoder_results = sess.run(auto_encoder.minimal_layer, feed_dict={x:train_dataset, auto_encoder.keep_prob: 1.0})

        print("New training shape", encoder_results.shape)

        encoder_results_valid = sess.run(auto_encoder.minimal_layer, feed_dict={x:valid_dataset, auto_encoder.keep_prob: 1.0})
        encoder_results_test = sess.run(auto_encoder.minimal_layer, feed_dict={x:test_dataset, auto_encoder.keep_prob: 1.0})
        with open("training_out.log", "w") as myfile:
            myfile.write("epoch,valid_acc,training_acc" + "\n")
        for epoch in range(epochs):

            # c = list(zip(encoder_results, train_labels))
            #
            # random.shuffle(c)
            #
            # encoder_results, train_labels = zip(*c)

            for i in range(0, total_len, batch_size):
                samples = slice(i, i + total_len)
                batch_x = encoder_results
                batch_y = train_labels

                sess.run(train_step, feed_dict={x_inp: batch_x, y: batch_y, keep_prob: 0.99,
                                                learning_rate: learn_rate,
                                                global_step: epoch})

            # print(sess.run(y_, feed_dict={x_inp: encoder_results_valid, keep_prob: 1.0})[0])

            learn_rate = max(learn_rate * learning_rate_decay, learning_rate_min)

            opt = str(epoch) + "," + \
                  str(sess.run(loss, feed_dict={x_inp: encoder_results_valid, y: valid_labels, keep_prob: 1.0,
                                                    global_step: epoch})) + \
                  "," + str(sess.run(loss, feed_dict={x_inp: encoder_results, y: train_labels, keep_prob: 1.0,
                                                          global_step: epoch}))



            with open("training_out.log", "a") as myfile:
                myfile.write(opt + "\n")

            if epoch % 1 == 0:
                print(opt)

            if epoch % 500 == 0:
                os.chdir(wd + "/model")
                save_path = saver.save(sess, str(epoch) + model_file)
                print("Model saved in file: %s" % save_path)
                os.chdir(wd)

        print("\nTraining complete!")
        writer.add_graph(sess.graph)

        opt = str(epoch) + "," + str(sess.run(loss, feed_dict={x_inp: encoder_results_test, y: test_labels,
                                                                   keep_prob: 1.0,
                                                                   global_step: epoch}))

        print("\r", opt)

        with open("training_out.log", "a") as myfile:
            myfile.write(opt + "\n")

        print()

        save_path = saver.save(sess, model_file)
        print("Model saved in file: %s" % save_path)

elif TRAIN_AUTOENCODER:
    # Auto-encoder
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        model_file = wd + "/model_encoding/model.ckpt"
        if os.path.isfile(wd + "/model_encoding/checkpoint"):
            saver_auto.restore(sess, model_file)
            print("Model restored.")

        total_len = train_labels.shape[0]

        count = 0

        learn_rate = learning_rate_start
        header = "epoch,valid_loss,train_loss,learn_rate"

        model_file = "_epoch/model.ckpt"

        with open("encoding_out.log", "w") as myfile:
            myfile.write(header + "\n")

        for epoch in range(epochs):
            c = list(zip(train_dataset, train_labels))

            random.shuffle(c)

            train_dataset, train_labels = zip(*c)

            for i in range(0, total_len, batch_size):
                samples = slice(i, i + total_len)
                batch_x = train_dataset[samples]
                batch_y = train_labels[samples]

                sess.run(train_auto_encoder_step, feed_dict={x: batch_x,
                                                             learning_rate: learn_rate,
                                                             auto_encoder.keep_prob: 0.9,
                                                             global_step: epoch})

            opt = str(epoch) + "," + str(sess.run(loss_auto_encoder, feed_dict={x: valid_dataset, auto_encoder.keep_prob: 1.0})) + \
                "," + str(sess.run(loss_auto_encoder, feed_dict={x: train_dataset, auto_encoder.keep_prob: 1.0})) + "," + str(learn_rate)

            learn_rate = max(learn_rate * learning_rate_decay, learning_rate_min)

            print(opt)

            with open("encoding_out.log", "a") as myfile:
                myfile.write(opt + "\n")

            if epoch % 100 == 0:
                os.chdir(wd + "/model_encoding")
                save_path = saver_auto.save(sess, str(epoch) + model_file)
                print("Model saved in file: %s" % save_path)
                os.chdir(wd)

            if epoch % 10 == 0:
                os.chdir(wd + "/model_encoding")
                save_path = saver_auto.save(sess, "tmp" + model_file)
                os.chdir(wd)

        print("\nTraining complete!")
        writer.add_graph(sess.graph)

        opt = str(epochs) + "," + str(sess.run(accuracy_auto_encoder, feed_dict={x: test_dataset, auto_encoder.keep_prob: 1.0}))

        print(opt)

        print()

        save_path = saver_auto.save(sess, model_file)
        print("Model saved in file: %s" % save_path)
