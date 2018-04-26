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

learning_rate = 0.001
epochs = 150000
batch_size = 30
percent_training = 0.7
percent_testing = 1
percent_validation = 0.5
hidden_layers = [4900, 4096, 256]
sd = 0.05
sdw = 0.05
show_output_image = True
image_height = 64

data = []
output = []
wd = os.getcwd()

if not os.path.exists(wd + "/model"):
    os.makedirs(wd + "/model")

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

os.chdir(wd)

data = np.array(data)

image_features = image_height * image_height

rem_features = data.shape[1] % image_features

output = np.array(output)

print("Data Shape:", data.shape, output.shape)

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
h_pool1 = tf.nn.tanh(max_pool_2x2(h_conv1))


W_conv2 = weight_variable([8, 8, 4, 8])
b_conv2 = bias_variable([8])

h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = tf.nn.tanh(max_pool_2x2(h_conv2))


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

print("model created successfully")

loss = tf.losses.mean_squared_error(y, y_)

#learning_rate = tf.placeholder(tf.float32, shape=[])

# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(learning_rate, 0.95, 1e-08, False).minimize(loss)


accuracy = tf.add(1.0, -tf.div(tf.reduce_mean(tf.losses.absolute_difference(y, y_)), 2.0))

tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf')

saver = tf.train.Saver()

plt.close('all')

plt.figure(1)

plots = 1

TRAIN_MODEL = True

# start the session
if TRAIN_MODEL:
    with tf.Session() as sess:

        print("Initialising Vars")
        sess.run(tf.global_variables_initializer())
        print("Done!")

        # Restore variables from disk.
        model_file = "model/model.ckpt"
        # if os.path.isfile("model/checkpoint"):
        #     saver.restore(sess, model_file)
        #     print("Model restored.")

        total_len = train_labels.shape[0]

        count = 0

        for epoch in range(epochs):
            samples = random.sample(range(total_len), batch_size)
            batch_x = train_dataset[samples]
            batch_y = train_labels[samples]
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.95})


            opt = str(epoch) + "," + str(sess.run(accuracy, feed_dict={x: valid_dataset, y: valid_labels, keep_prob: 1.0}))

            print(opt)



            with open("training_out.log", "a") as myfile:
                myfile.write(opt + "\n")



            if epoch % 50 == 0:
                os.chdir(wd + "/model")
                save_path = saver.save(sess, str(epoch) + model_file)
                print("Model saved in file: %s" % save_path)
                os.chdir(wd)

        print("\nTraining complete!")
        writer.add_graph(sess.graph)

        opt = "-1," + str(sess.run(accuracy, feed_dict={x: test_dataset, y: test_labels, keep_prob: 1.0}))

        print(opt)

        with open("training_out.log", "a") as myfile:
            myfile.write(opt + "\n")

        print()

        save_path = saver.save(sess, model_file)
        print("Model saved in file: %s" % save_path)