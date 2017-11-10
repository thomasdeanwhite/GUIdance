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
from sklearn import decomposition
from sklearn import preprocessing


def weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=1)
    return tf.Variable(initial)

class AutoEncoder:
    learning_rate = 0.0001
    epochs = 1000
    batch_size = 25
    input_features = 64*64+4
    image_size = 64*64
    hidden_layers_n = [24*24, 16*16]
    image = None
    hidden_layers = [None]
    output_layer = None
    minimal_layer = None
    loss = None

    def __init__(self, image=tf.placeholder(tf.float32, [None, image_size])):
        self.image = image

        hidden_layers_n = [self.image_size]
        hidden_layers_n.extend(self.hidden_layers_n)

        self.hidden_layers = []
        last_layer = self.image
        last_size = 0

        input_stack = []

        for i in range(0, len(hidden_layers_n)-1):
            weights = weight_variable([hidden_layers_n[i], hidden_layers_n[i+1]])
            biases = bias_variable([hidden_layers_n[i+1]])
            hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(last_layer, weights), biases))
            last_layer = hidden_layer
            self.hidden_layers.append(hidden_layer)
            input_stack.append(hidden_layers_n[i])
            last_size = hidden_layers_n[i+1]

        self.minimal_layer = last_layer

        while len(input_stack) > 0:
            next_layer = input_stack.pop()
            prev_layer = last_size
            weights = weight_variable([prev_layer, next_layer])
            biases = bias_variable([next_layer])
            hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(last_layer, weights), biases))
            last_layer = hidden_layer
            self.hidden_layers.append(hidden_layer)
            last_size = next_layer

        last_layer = self.hidden_layers.pop()
        self.output_layer = last_layer

        self.loss = tf.reduce_mean(tf.pow(self.image - self.output_layer, 2))