import config as cfg
from yolo import Yolo
from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import sys
import gc
import math
import random
import os
import pickle
import re

tf.logging.set_verbosity(tf.logging.INFO)

def normalise_point(point, val):
    i = point*val
    return i, math.floor(i)

def normalise_label(label):
    px, cx = normalise_point(max(0, min(1, label[0])), cfg.grid_shape[0])
    py, cy = normalise_point(max(0, min(1, label[1])), cfg.grid_shape[1])
    return [
        px,
        py,
        max(0, min(cfg.grid_shape[0], label[2]*cfg.grid_shape[0])),
        max(0, min(cfg.grid_shape[1], label[3]*cfg.grid_shape[1])),
        label[4]
    ], (cx, cy)

def load_files(raw_files):
    raw_files = [f.replace("/data/acp15tdw", "/home/thomas/experiments") for f in raw_files]
    label_files = [f.replace("/images/", "/labels/") for f in raw_files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    pickle_files = [f.replace("/images/", "/pickle/") for f in raw_files]
    pickle_files = [f.replace(".png", ".pickle") for f in pickle_files]

    images = []
    labels = []
    object_detection = []

    for i in range(len(raw_files)):
        pickle_f = pickle_files[i]


        pickled_data = []

        if os.path.isfile(pickle_f):
            pickled_data = pickle.load(open(pickle_f, "rb"))
            images.append(pickled_data[0])
            labels.append(pickled_data[1])
            object_detection.append(pickled_data[2])
        else:
            f = raw_files[i]
            f_l = label_files[i]
            image = np.int16(imread(f, 0))

            image = np.uint8(resize(image, (cfg.width, cfg.height)))
            image = np.reshape(image, [cfg.width, cfg.height, 1])


            image = np.reshape(image, [cfg.width, cfg.height, 1])
            images.append(image)

            # read in format [c, x, y, width, height]
            # store in format [c], [x, y, width, height]
            with open(f_l, "r") as l:
                obj_detect = [[0 for i in
                               range(cfg.grid_shape[0])]for i in
                              range(cfg.grid_shape[1])]
                imglabs = [[[0 for i in
                             range(5)]for i in
                            range(cfg.grid_shape[1])] for i in
                           range(cfg.grid_shape[0])]

                for line in l:
                    elements = line.split(" ")
                    #print(elements[1:3])
                    normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                                float(elements[3]), float(elements[4]), 1])
                    x = max(0, min(int(centre[0]), cfg.grid_shape[0]-1))
                    y = max(0, min(int(centre[1]), cfg.grid_shape[1]-1))
                    imglabs[y][x] = normalised_label
                    obj_detect[y][x] = int(elements[0])
                    #obj_detect[y][x][int(elements[0])] = 1

                object_detection.append(obj_detect)
                labels.append(imglabs)

            pickled_data = [image, imglabs, obj_detect]

            pickle.dump(pickled_data, open(pickle_f, "wb"))

    return images, labels, object_detection

if __name__ == '__main__':

    training_file = cfg.data_dir + "/" + cfg.train_file

    valid_images = []

    real_images = []

    pattern = re.compile(".*\/([0-9]+).*")

    with open(training_file, "r") as tfile:
        for l in tfile:

            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())



    valid_file = cfg.data_dir + "/" + cfg.validate_file

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    valid_file = cfg.data_dir + "/" + cfg.test_file

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    valid_file = cfg.data_dir + "/test-balanced.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num > 243:
                valid_images.append(l.strip())

    #valid_images = random.sample(valid_images, cfg.batch_size)

    with tf.device(cfg.gpu):

        tf.reset_default_graph()

        yolo = Yolo()

        yolo.create_network()

        yolo.set_training(True)

        yolo.create_training()

        learning_rate = tf.placeholder(tf.float64)
        learning_r = cfg.learning_rate_start
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            yolo.set_update_ops(update_ops)
            train_step = tf.train.MomentumOptimizer(learning_rate, cfg.momentum). \
                minimize(yolo.loss)

        saver = tf.train.Saver()

        model_file = cfg.weights_dir + "/model.ckpt"

        cfg.batch_size = 1

        valid_batches = int(len(valid_images) / cfg.batch_size)

        real_batches = int(len(real_images) / cfg.batch_size)

        gpu_options = tf.GPUOptions(allow_growth=True)

        config = tf.ConfigProto(allow_soft_placement = True, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:

            init_op = tf.global_variables_initializer()

            print("Initialising Memory Values")
            model = sess.run(init_op)

            #print(tf.get_default_graph().as_graph_def())

            if os.path.isfile(os.getcwd() + "/" + cfg.weights_dir + "/checkpoint"):
                saver.restore(sess, model_file)
                print("Restored model")
            else:
                print("Training from scratch.")

            if (cfg.enable_logging):
                train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)

            anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
            print("anchors", anchors.shape)

            random.shuffle(valid_images)
            header_string = "threshold,dataset,image,var,val"
            with open("validation.csv", "w") as file:
                file.write(header_string + "\n")

            print(header_string)

            yolo.set_training(False)

            header_vals = ["precision", "recall", "mAP"]

            for i in range(1):

                values = [0, 0, 0]

                for j in range(valid_batches):
                    gc.collect()
                    lower_index = j
                    upper_index = j+1

                    v_imgs, v_labels, v_obj_detection = load_files(
                        valid_images[lower_index:upper_index])

                    v_imgs = (np.array(v_imgs)/127.5)-1

                    v_labels = np.array(v_labels)

                    v_obj_detection = np.array(v_obj_detection)

                    cfg.object_detection_threshold = 0.3

                    if len(v_labels) == 0:
                        continue

                    true_pos, false_pos, false_neg, mAP = sess.run([
                        yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: 0.5,
                        yolo.object_detection_threshold: 0.3
                    })
                    values[0] = 100*(true_pos/(true_pos+false_pos))
                    values[1] = 100*(true_pos/(true_pos+false_neg))
                    values[2] = 100*mAP

                    sens_string = ""

                    for acc in range(len(values)):
                        sens_string += "\n" + str("%.2f" % (i*0.05)) + ",synthetic," + str(j) + ","
                        sens_string += str(header_vals[acc]) + ","
                        sens_string += str(values[acc])

                    print(sens_string)
                    with open("validation.csv", "a") as file:
                        file.write(sens_string + "\n")

                values = [0, 0, 0]

                for j in range(real_batches):
                    gc.collect()
                    lower_index = j
                    upper_index = j+1

                    v_imgs, v_labels, v_obj_detection = load_files(
                        real_images[lower_index:upper_index])

                    v_imgs = (np.array(v_imgs)/127.5)-1

                    v_labels = np.array(v_labels)

                    v_obj_detection = np.array(v_obj_detection)

                    cfg.object_detection_threshold = 0.3

                    if len(v_labels) == 0:
                        continue

                    true_pos, false_pos, false_neg, mAP = sess.run([
                        yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: 0.5,
                        yolo.object_detection_threshold: 0.3
                    })
                    values[0] = 100*(true_pos/(true_pos+false_pos))
                    values[1] = 100*(true_pos/(true_pos+false_neg))
                    values[2] = 100*mAP

                    sens_string = ""

                    for acc in range(len(values)):
                        sens_string += "\n" + str("%.2f" % (i*0.05)) + ",real," + str(j) + ","
                        sens_string += str(header_vals[acc]) + ","
                        sens_string += str(values[acc])

                    print(sens_string)
                    with open("validation.csv", "a") as file:
                        file.write(sens_string + "\n")


            gc.collect()

            sys.exit()