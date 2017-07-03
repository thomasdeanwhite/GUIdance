# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
from six.moves import cPickle as pickle



learning_rate = 0.01
epochs = 10
batch_size = 128
letters = 10
image_size = 28
num_labels = 10

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    # print('Training set', train_dataset.shape, train_labels.shape)
    # print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



# def accuracy(predictions, labels):
#     return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
#             / predictions.shape[0])



# Python optimisation variables
learning_rate = 0.01
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 150], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([150]), name='b2')

# and the weights connecting the hidden layer to the output layer
W3 = tf.Variable(tf.random_normal([150, 75], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([75]), name='b3')

# and the weights connecting the hidden layer to the output layer
W4 = tf.Variable(tf.random_normal([75, 10], stddev=0.03), name='W4')
b4 = tf.Variable(tf.random_normal([10]), name='b4')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
#hidden_out = tf.nn.relu(hidden_out)

# calculate the output of the hidden layer
hidden_out2 = tf.add(tf.matmul(hidden_out, W2), b2)
#hidden_out2 = tf.nn.relu(hidden_out2)

# calculate the output of the hidden layer
hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)
hidden_out3 = tf.nn.relu(hidden_out3)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out3, W4), b4))

# now let's define the cost function which we are going to train the model on
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                              + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# add a summary to store the accuracy
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf')
# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_len = len(train_labels)
    total_batch = int(total_len / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            samples = random.sample(range(total_len), batch_size)
            batch_x = train_dataset[samples]
            batch_y = train_labels[samples]
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        summary = sess.run(merged, feed_dict={x: valid_dataset, y: valid_labels})
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: test_dataset, y: test_labels}))

