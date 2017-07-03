# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import csv
import os
from six.moves import cPickle as pickle


learning_rate = 0.0001
epochs = 30
batch_size = 50
percent_training = 0.7
percent_testing = 1
percent_validation = 0.5

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
output = np.array(output)

print("Records:", data.shape[0])

output = np.array(output)


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

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([train_dataset.shape[1], 1024], stddev=0.05), name='W1')
b1 = tf.Variable(tf.random_normal([1024]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([1024, 128], stddev=0.05), name='W2')
b2 = tf.Variable(tf.random_normal([128]), name='b2')

# and the weights connecting the hidden layer to the output layer
W3 = tf.Variable(tf.random_normal([128, 64], stddev=0.05), name='W3')
b3 = tf.Variable(tf.random_normal([64]), name='b3')

# and the weights connecting the hidden layer to the output layer
W4 = tf.Variable(tf.random_normal([64, train_labels.shape[1]], stddev=0.05), name='W4')
b4 = tf.Variable(tf.random_normal([train_labels.shape[1]]), name='b4')

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


loss = tf.contrib.losses.absolute_difference(y, y_)
# add an optimiser
# optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# add a summary to store the accuracy
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf')

saver = tf.train.Saver()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    # Restore variables from disk.
    model_file = "model/model.ckpt"
    if os.path.isfile("model/checkpoint"):
        saver.restore(sess, model_file)
        print("Model restored.")


    total_len = train_labels.shape[0]
    total_batch = int(total_len / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            samples = random.sample(range(total_len), batch_size)
            batch_x = train_dataset[samples]
            batch_y = train_labels[samples]
            _, c = sess.run([optimiser, loss], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Valid Accuracy: ", sess.run(accuracy, feed_dict={x: valid_dataset, y: valid_labels}), "cost:", avg_cost)
        summary = sess.run(merged, feed_dict={x: valid_dataset, y: valid_labels})
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: test_dataset, y: test_labels}))

    save_path = saver.save(sess, model_file)
    print("Model saved in file: %s" % save_path)


