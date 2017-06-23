from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import urllib.request as urllib2

import numpy as np
import tensorflow as tf
import csv
import random

# Data sets
TRAINING_INPUTS = "C:/work/NuiMimic/NuiMimic/default-user/training_inputs.csv"
TRAINING_OUTPUTS = "C:/work/NuiMimic/NuiMimic/default-user/training_outputs.csv"


def main():
    # Load datasets.
    # training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    #     filename="iris_training.csv",
    #     target_dtype=np.int,
    #     features_dtype=np.float32)

    # print(training_set.data)

    training_set_inp = []
    training_set_out = []

    testing_set_inp = []
    testing_set_out = []

    with open(TRAINING_INPUTS) as t_inputs:
        readr = csv.reader(t_inputs, delimiter=',', quotechar='"')
        for row in readr:
            training_set_inp.append(row)

    with open(TRAINING_OUTPUTS) as t_outputs:
        readr = csv.reader(t_outputs, delimiter=',', quotechar='"')
        for row in readr:
            training_set_out.append(row)

    del training_set_inp[0]

    dims = len(training_set_inp[0]);
    outputs = len(training_set_out[0])

    print(str(len(training_set_out)) + " " + str(len(training_set_inp)))

    for i in range(0, len(training_set_out)):
        training_set_out[i] = [float(k) for k in
                               training_set_out[i]]
        training_set_inp[i] = [float(k) for k in
                               training_set_inp[i]]

    # for i in range(0, len(training_set_inp)):
    #     training_set_inp[i] = tf.constant(training_set_inp[i])

    testing_sample = sorted(random.sample(range(0, len(training_set_inp) - 1),
                                          int(len(training_set_inp) / 5)),
                            reverse=True)

    print(testing_sample)

    for i in testing_sample:
        testing_set_inp.append(training_set_inp[i])
        testing_set_out.append(training_set_out[i])
        del training_set_inp[i]
        del training_set_out[i]

    training_set_out = tf.constant(training_set_out, tf.float32)
    training_set_inp = tf.constant(training_set_inp, tf.float32)

    testing_set_out = tf.constant(testing_set_out, tf.float32)
    testing_set_inp = tf.constant(testing_set_inp, tf.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("",
                                                            dimension=dims)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=outputs,
                                                model_dir="iris_model")

    # Define the training inputs
    def get_train_inputs():
        x = training_set_inp
        y = training_set_out

        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Define the test inputs
    def get_test_inputs():
        x = testing_set_inp
        y = testing_set_out

        return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == "__main__":
    main()
