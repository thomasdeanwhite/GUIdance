import config as cfg
from yolo import Yolo
from cv2 import imread, resize
import cv2
import numpy as np
import tensorflow as tf
import sys
import gc
import math
import random
import os
import pickle
import re
from data_loader import load_files

debug = False

last_epochs = 12

tf.logging.set_verbosity(tf.logging.INFO)

def count_params():
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])


if __name__ == '__main__':

    tf.reset_default_graph()
    yolo = Yolo()

    yolo.create_network()
    #yolo.set_training(False)
    #yolo.create_training()

    learning_rate = tf.placeholder(tf.float64)
    learning_r = cfg.learning_rate_start

    saver = tf.train.Saver()

    model_file = os.getcwd() + "/" + cfg.weights_dir + "/model.ckpt"

    #chkp.print_tensors_in_checkpoint_file(model_file, tensor_name='', all_tensors=True)

    gpu_options = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:

        model_file = "weights-full" + "/model.ckpt"

        init_op = tf.global_variables_initializer()

        print("Initialising Memory Values")
        model = sess.run(init_op)

        #print(tf.get_default_graph().as_graph_def())

        if os.path.isfile(os.getcwd() + "/" + "weights-full"  + "/checkpoint"):
            saver.restore(sess, model_file)
            print("Restored model")
        else:
            print("Training from scratch.")

        reader = tf.train.NewCheckpointReader("weights-full" + "/model.ckpt")

        print('\nCount the number of parameters in ckpt file(%s)' % cfg.weights_dir)
        param_map = reader.get_variable_to_shape_map()
        total_count = 0
        for k, v in param_map.items():
            if 'Adam' not in k and 'global_step' not in k:
                temp = np.prod(v)
                total_count += temp
                print('%s: %s => %d' % (k, str(v), temp))

        vars = tf.trainable_variables()
        print(vars) #some infos about variables...
        vars_vals = sess.run(vars)
        count = 0

        positive = 0
        for var, val in zip(vars, vars_vals):
            print("var: {}, value: {}".format(var.name, val))
            count += val.size
            positive += np.sum((val >= 0).astype(int))

        print('Total Param Count: %d' % total_count)
        print("Activations", positive, count)

