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

totals = []

for i in range(10):
    totals.append(0)

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
                totals[int(elements[0])] += 1
                #obj_detect[y][x][int(elements[0])] = 1

            object_detection.append(obj_detect)
            labels.append(imglabs)



    return images, labels, object_detection

if __name__ == '__main__':

    training_file = cfg.data_dir + "/train.txt"

    valid_images = []

    real_images = []

    pattern = re.compile(".*\/([0-9]+).*")

    with open(training_file, "r") as tfile:
        for l in tfile:

            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())



    valid_file = cfg.data_dir + "/validate.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    valid_file = cfg.data_dir + "/test.txt"

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

    valid_images = valid_images[:150]

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

        print("Batches:", valid_batches, real_batches)

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
            header_string = "iou_threshold,dataset,class,rank,correct,precision,recall,confidence,iou"
            with open("validation.csv", "w") as file:
                file.write(header_string + "\n")

            yolo.set_training(False)

            iou_threshold = 0.2
            confidence_threshold = -1

            class_predictions = []

            class_totals = []

            for i in range(1):

                for j in range(valid_batches):
                    gc.collect()
                    lower_index = j
                    upper_index = j+1

                    v_imgs, v_labels, v_obj_detection = load_files(
                        valid_images[lower_index:upper_index])

                    v_imgs = (np.array(v_imgs)/127.5)-1

                    v_labels = np.array(v_labels)

                    v_obj_detection = np.array(v_obj_detection)

                    cfg.object_detection_threshold = confidence_threshold

                    if len(v_labels) == 0:
                        continue

                    res, correct, iou = sess.run([
                        yolo.output, yolo.matches, yolo.best_iou], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: iou_threshold,
                        yolo.object_detection_threshold: confidence_threshold
                    })

                    labels = yolo.convert_net_to_bb(res, filter_top=True)

                    for rc in range(len(yolo.names)):
                        if (len(class_predictions) < rc + 1):
                            class_predictions.append([])
                        if (len(class_totals) < rc + 1):
                            class_totals.append(0)

                        cl_equals = np.where(np.equal(res[..., 4], 1), np.equal(v_obj_detection, np.zeros_like(v_obj_detection)+rc), 0)

                        cl_quantity = np.sum(cl_equals.astype(np.int32))

                        if cl_quantity > 0:
                            class_totals[rc] += cl_quantity

                            for ic in range(cfg.grid_shape[0]):
                                for jc in range(cfg.grid_shape[1]):

                                    label = labels[(jc*cfg.grid_shape[0]) + ic]

                                    if label[0] == rc and label[5] > confidence_threshold:
                                        class_predictions[rc].append([labels[(jc*cfg.grid_shape[1]) + ic][5], correct[0][ic][jc], iou[0][ic][jc][0][0]])

                    del v_imgs
                    del v_labels
                    del v_obj_detection

                for rc in range(len(class_predictions)):

                    if (class_totals[rc] == 0):
                        class_totals[rc] = 1

                    class_predictions[rc] = sorted(class_predictions[rc], key=lambda box: -box[0])
                    correct_n = 0
                    for box in range(len(class_predictions[rc])):

                        correct_n += class_predictions[rc][box][1]

                        #assert correct_n <= class_totals[rc]

                        sens_string = str(iou_threshold) + ",synthetic," + yolo.names[rc] + "," + str(box+1) + "," + str(class_predictions[rc][box][1]) + "," + \
                        str(correct_n / (box+1)) + "," + str(correct_n/totals[rc]) + "," + str(class_predictions[rc][box][0]) + "," + str(class_predictions[rc][box][2]) + "\n"

                        with open("validation.csv", "a") as file:
                            file.write(sens_string + "\n")

                print(totals)

                class_predictions = []

                class_totals = []

                totals = []

                for i in range(10):
                    totals.append(0)

                for j in range(real_batches):
                    gc.collect()
                    lower_index = j
                    upper_index = j+1

                    v_imgs, v_labels, v_obj_detection = load_files(
                        real_images[lower_index:upper_index])

                    v_imgs = (np.array(v_imgs)/127.5)-1

                    v_labels = np.array(v_labels)

                    v_obj_detection = np.array(v_obj_detection)

                    cfg.object_detection_threshold = confidence_threshold

                    if len(v_labels) == 0:
                        continue

                    res, correct, iou = sess.run([
                        yolo.output, yolo.matches, yolo.best_iou], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: iou_threshold,
                        yolo.object_detection_threshold: confidence_threshold
                    })

                    labels = yolo.convert_net_to_bb(res, filter_top=True)

                    for rc in range(len(yolo.names)):
                        if (len(class_predictions) < rc + 1):
                            class_predictions.append([])
                        if (len(class_totals) < rc + 1):
                            class_totals.append(0)

                        cl_equals = np.where(np.equal(res[..., 4], 1), np.equal(v_obj_detection, np.zeros_like(v_obj_detection)+rc), 0)
                        cl_quantity = np.sum(cl_equals.astype(np.int32))

                        if cl_quantity > 0:

                            label = labels[(jc*cfg.grid_shape[0]) + ic]

                            for ic in range(cfg.grid_shape[0]):
                                for jc in range(cfg.grid_shape[1]):
                                    if label[0] == rc and label[5] > confidence_threshold:
                                        class_predictions[rc].append([labels[(jc*cfg.grid_shape[1]) + ic][5], correct[0][ic][jc], iou[0][ic][jc][0][0]])
                    del v_imgs
                    del v_labels
                    del v_obj_detection

                for rc in range(len(class_predictions)):

                    if (class_totals[rc] == 0):
                        class_totals[rc] = 1

                    class_predictions[rc] = sorted(class_predictions[rc], key=lambda box: -box[0])
                    correct_n = 0
                    for box in range(len(class_predictions[rc])):

                        correct_n += class_predictions[rc][box][1]

                        sens_string = str(iou_threshold) + ",real," + yolo.names[rc] + "," + str(box+1) + "," + str(class_predictions[rc][box][1]) + "," + \
                                      str(correct_n / (box+1)) + "," + str(correct_n/totals[rc]) + "," + str(class_predictions[rc][box][0]) + "," + str(class_predictions[rc][box][2]) + "\n"

                        with open("validation.csv", "a") as file:
                            file.write(sens_string + "\n")

            print(totals)


            gc.collect()

            sys.exit()