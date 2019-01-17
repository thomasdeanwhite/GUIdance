import config as cfg
from yolo import Yolo
import cv2
import numpy as np
import tensorflow as tf
import sys
import gc
import math
import random
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
from data_loader import load_files, load_raw_image, disable_transformation
import data_loader
import re

def convert_coords(x, y, w, h, aspect):
    if aspect > 1: # width is bigger than height
        h = h * aspect
        y = 0.5 + ((y - 0.5)*aspect)
    elif aspect < 1:
        w = w / aspect
        x = 0.5 + ((x - 0.5)/aspect)

    return x, y, w, h

if __name__ == '__main__':

    data_loader.debug = True

    training_file = cfg.data_dir + "/train-25000.txt"

    pattern = re.compile(".*\/([0-9]+).*")

    training_images = []

    real_images = []

    with open(training_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])
            training_images.append(l.strip())

    training_images = [f.replace("/home/thomas/work/GuiImages/public", "/data/acp15tdw/data") for f in training_images]

    while len(training_images) < 100000:
        training_images = training_images + training_images

    training_images = training_images[:100000]

    valid_file = cfg.data_dir + "/" + cfg.validate_file

    valid_images = []

    with open(valid_file, "r") as tfile:
        for l in tfile:
            valid_images.append(l.strip())

    random.shuffle(valid_images)
    valid_images = valid_images[:500]

    training_file = cfg.data_dir + "/../backup/data/train.txt"

    with open(training_file, "r") as tfile:
        for l in tfile:

            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())



    valid_file = cfg.data_dir + "/../backup/data/validate.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    valid_file = cfg.data_dir + "/../backup/data/test.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    real_images = [f.replace("/data/acp15tdw", "/data/acp15tdw/backup") for f in real_images]

    real_images_manual = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(cfg.data_dir + "/../real/data/images")) for f in fn]

    for ri in real_images_manual:
        real_images.append(ri)

    # random.seed(cfg.random_seed)
    # random.shuffle(real_images)
    #training_images = real_images[:100]

    #real_images = real_images[100:]

    print("Found", len(real_images), "real GUI screenshots.")

    #valid_images = random.sample(valid_images, cfg.batch_size)

    tf.reset_default_graph()

    yolo = Yolo()

    yolo.create_network()

    yolo.set_training(True)

    yolo.create_training()

    global_step = tf.placeholder(tf.int32)
    batches = math.ceil(len(training_images)/cfg.batch_size) if cfg.run_all_batches else 1


    learning_rate = tf.train.exponential_decay(0.01, global_step,
                                               1, 0.9, staircase=True)
    #learning_rate = tf.placeholder(tf.float64)
    #learning_r = cfg.learning_rate_start
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        yolo.set_update_ops(update_ops)

        train_step = tf.train.AdamOptimizer(5e-5). \
            minimize(yolo.loss)

    saver = tf.train.Saver()

    model_file = cfg.weights_dir + "/model.ckpt"

    valid_batches = math.ceil(len(valid_images)/cfg.batch_size) if cfg.run_all_batches else 1

    real_batches = math.ceil(len(real_images)/cfg.batch_size) if cfg.run_all_batches else 1

    if (real_batches < 1):
        real_batches = 1

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:

        init_op = tf.global_variables_initializer()

        print("Initialising Memory Values")
        model = sess.run(init_op)

        random.shuffle(valid_images)

        random.shuffle(training_images)
        yolo.set_training(False)

        losses = [0, 0, 0, 0, 0, 0, 0, 0]

        for j in range(valid_batches):
            gc.collect()

            lower_index = j*cfg.batch_size
            upper_index = min(len(valid_images), ((j+1)*cfg.batch_size))

            v_imgs, v_labels, v_obj_detection = load_files(
                valid_images[lower_index:upper_index])

            v_imgs = (np.array(v_imgs)/127.5)-1

            v_labels = np.array(v_labels)

            v_obj_detection = np.array(v_obj_detection)