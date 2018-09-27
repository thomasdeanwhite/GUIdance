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
from data_loader import load_files, disable_transformation

tf.logging.set_verbosity(tf.logging.INFO)

totals = []

disable_transformation()

if __name__ == '__main__':

    confusion = []

    training_file = cfg.data_dir + "/../backup/data/train.txt"

    valid_images = []

    real_images = []

    pattern = re.compile(".*\/([0-9]+).*")

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

    real_images = [f.replace("/data/acp15tdw", "/data/acp15tdw/backup") for f in real_images]

    valid_file = cfg.data_dir + "/../backup/data/test.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    valid_file = cfg.data_dir + "/test.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])
            valid_images.append(l.strip())

    #valid_images = random.sample(valid_images, cfg.batch_size)

    #valid_images = valid_images[:1500]

    with tf.device(cfg.gpu):

        tf.reset_default_graph()

        yolo = Yolo()

        yolo.create_network()

        yolo.set_training(True)

        yolo.create_training()

        for i in range(len(yolo.names)):
            totals.append(0)

        for x in range(len(yolo.names)):
            confus = []
            for y in range(len(yolo.names)):
                confus.append(0)
            confusion.append(confus)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        batches = math.ceil(len(valid_images)/cfg.batch_size) if cfg.run_all_batches else 1


        learning_rate = tf.train.exponential_decay(0.1, global_step,
                                                   batches, 0.9, staircase=True)
        #learning_rate = tf.placeholder(tf.float64)
        #learning_r = cfg.learning_rate_start
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            yolo.set_update_ops(update_ops)

            train_step = tf.train.AdadeltaOptimizer(learning_rate). \
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

            iou_threshold = 0.5
            confidence_threshold = 0.01

            class_predictions = []

            class_totals = []

            for i in range(1):

                iou_threshold = 0.5 + (i * 0.05)

                pred_errors = []

                for i in range(len(yolo.names)):
                    pred_errors.append([0, 0])

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

                    #v_obj_detection = np.zeros_like(v_obj_detection)

                    o_img, o_h, o_w, = res.shape[:3]

                    img = o_img-1

                    while img >= 0:
                        h = o_h-1
                        while h >= 0:
                            w = o_w-1
                            while w >= 0:
                                lab = res[img,h,w]
                                if v_labels[img,h,w,4] > 0:
                                    clazz = np.argmax(lab[25:])
                                    c_clazz = v_obj_detection[img,h,w]
                                    confusion[clazz][c_clazz] += 1
                                    totals[c_clazz] += 1
                                w = w-1
                            h = h-1
                        img = img-1

                    for rc in range(len(yolo.names)):
                        if (len(class_predictions) < rc + 1):
                            class_predictions.append([])
                        if (len(class_totals) < rc + 1):
                            class_totals.append(0)

                        cl_equals = np.where(res[..., 4] > confidence_threshold, np.equal(v_obj_detection, np.zeros_like(v_obj_detection)+rc), 0)

                        cl_quantity = np.sum(cl_equals.astype(np.int32))

                        if cl_quantity > 0:
                            class_totals[rc] += cl_quantity

                            for ic in range(cfg.grid_shape[0]):
                                for jc in range(cfg.grid_shape[1]):

                                    label = labels[(jc*cfg.grid_shape[0]) + ic]

                                    if label[5] > confidence_threshold and int(label[0]) == rc:
                                        class_predictions[rc].append([labels[(jc*cfg.grid_shape[1]) + ic][5], correct[0][ic][jc], iou[0][ic][jc][0][0]])

                    del v_imgs
                    del v_labels
                    del v_obj_detection

                #print(confusion)

                for rc in range(len(class_predictions)):

                    if (class_totals[rc] == 0):
                        class_totals[rc] = 1

                    class_predictions[rc] = sorted(class_predictions[rc], key=lambda box: -box[0])
                    correct_n = 0

                    total = totals[rc]

                    for box in range(len(class_predictions[rc])):

                        correct_n += class_predictions[rc][box][1]

                        #assert correct_n <= class_totals[rc]



                        sens_string = str(iou_threshold) + ",synthetic," + yolo.names[rc] + "," + str(box+1) + "," + str(class_predictions[rc][box][1]) + "," + \
                                      str(correct_n / (box+1)) + "," + str(correct_n/total) + "," + str(class_predictions[rc][box][0]) + "," + str(class_predictions[rc][box][2]) + "\n"

                        with open("validation.csv", "a") as file:
                            file.write(sens_string + "\n")

                print(totals)

                confusion_s = "predicted_class,actual_class,quantity,dataset,total\n"

                for x in range(len(yolo.names)):
                    for y in range(len(yolo.names)):
                        confusion_s += yolo.names[x] + "," + yolo.names[y] + "," + str(confusion[x][y]) + ",synthetic," + str(totals[y]) + "\n"

                with open("confusion.csv", "w") as file:
                    file.write(confusion_s + "\n")


                confusion = []

                for x in range(len(yolo.names)):
                    confus = []
                    for y in range(len(yolo.names)):
                        confus.append(0)
                    confusion.append(confus)

                class_predictions = []

                class_totals = []

                totals = []

                for i in range(len(yolo.names)):
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

                    img, h, w, = res.shape[:3]

                    img -= 1
                    h -= 1
                    w -= 1

                    o_img, o_h, o_w, = res.shape[:3]

                    img = o_img-1

                    while img >= 0:
                        h = o_h-1
                        while h >= 0:
                            w = o_w-1
                            while w >= 0:
                                lab = res[img,h,w]
                                if v_labels[img,h,w,4] > 0:
                                    clazz = np.argmax(lab[25:])
                                    c_clazz = v_obj_detection[img,h,w]
                                    confusion[clazz][c_clazz] += 1
                                    totals[c_clazz] += 1
                                w = w-1
                            h = h-1
                        img = img-1

                    #v_obj_detection = np.zeros_like(v_obj_detection)

                    for rc in range(len(yolo.names)):
                        if (len(class_predictions) < rc + 1):
                            class_predictions.append([])
                        if (len(class_totals) < rc + 1):
                            class_totals.append(0)

                        cl_equals = np.where(res[..., 4] > confidence_threshold, np.equal(v_obj_detection, np.zeros_like(v_obj_detection)+rc), 0)

                        cl_quantity = np.sum(cl_equals.astype(np.int32))

                        if cl_quantity > 0:
                            class_totals[rc] += cl_quantity

                            for ic in range(cfg.grid_shape[0]):
                                for jc in range(cfg.grid_shape[1]):

                                    label = labels[(jc*cfg.grid_shape[0]) + ic]

                                    if label[5] > confidence_threshold and int(label[0]) == rc:
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

            confusion_s = ""

            for x in range(len(yolo.names)):
                for y in range(len(yolo.names)):
                    confusion_s += yolo.names[x] + "," + yolo.names[y] + "," + str(confusion[x][y]) + ",real," + str(totals[y]) + "\n"

            with open("confusion.csv", "a") as file:
                file.write(confusion_s + "\n")


            confusion = []

            print(totals)


            gc.collect()

sys.exit()