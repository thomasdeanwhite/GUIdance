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
    #raw_files = [f.replace("/data/acp15tdw", "/home/thomas") for f in raw_files]
    files = []
    #for f in raw_files:
    #    if os.path.isfile(f):
    #        files.append(f)

    images = []
    labels = []
    object_detection = []

    for f in raw_files:
        if f == None or not os.path.isfile(f):
            continue
        raw_img = imread(f, 0)
        if raw_img is None:
            continue

        files.append(f)
        image = np.int16(raw_img)

        if random.random() < cfg.brightness_probability:
            brightness = int(random.random()*cfg.brightness_var*2)-cfg.brightness_var
            image = np.maximum(0, np.minimum(255, np.add(image, brightness)))

        if random.random() < cfg.contrast_probability:
            contrast = (random.random() * cfg.contrast_var * 2) - cfg.contrast_var

            contrast_diff = (image - np.mean(image)) * contrast
            image = np.maximum(0, np.minimum(255, np.add(image, contrast_diff)))

        if random.random() < cfg.invert_probability:
            image = 255 - image

        image = np.uint8(resize(image, (cfg.width, cfg.height)))
        image = np.reshape(image, [cfg.width, cfg.height, 1])


        image = np.reshape(image, [cfg.width, cfg.height, 1])
        images.append(image)

    label_files = [f.replace("/images/", "/labels/") for f in files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    for f in label_files:
        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]
        with open(f, "r") as l:
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
    return images, labels, object_detection

def modify_learning_rate(epoch):
    # return learning rate in accordance to YOLO paper
    # if epoch == 0:
    #     return 0.001
    # if epoch < 10:
    #     return 0.001+(0.01-0.001)/((10-epoch))
    #
    # if epoch < 75:
    #     return 0.01
    #
    # if epoch < 105:
    #     return 0.001
    #
    # return 0.0001

    if epoch < 5: # temp epoch start at 70
        return 0.01
    if epoch < 35:
        return 0.001

    return 0.0001


if __name__ == '__main__':

    training_file = cfg.data_dir + "/" + cfg.train_file

    training_images = []

    with open(training_file, "r") as tfile:
        for l in tfile:
            training_images.append(l.strip())



    valid_file = cfg.data_dir + "/" + cfg.validate_file

    valid_images = []

    with open(valid_file, "r") as tfile:
        for l in tfile:
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

        valid_batches = math.ceil(len(valid_images)/cfg.batch_size) if cfg.run_all_batches else 1

        config = tf.ConfigProto(allow_soft_placement = True)

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

            print("!Finished Initialising Memory Values!")
            image_length = len(training_images)
            batches = math.ceil(image_length/cfg.batch_size) if cfg.run_all_batches else 1
            print("Starting training:", image_length, "images in", batches, "batches.")

            anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
            print("anchors", anchors.shape)

            random.shuffle(valid_images)
            with open("training.csv", "w") as file:
                file.write("epoch,dataset,loss,loss_position,loss_dimension,loss_obj,loss_class,precision,recall,mAP\n")

            for i in range(cfg.epochs):
                #random.shuffle(training_images)
                yolo.set_training(False)

                losses = [0, 0, 0, 0, 0, 0, 0, 0]

                for j in range(valid_batches):
                    gc.collect()
                    print("\rValidating " + str(j) + "/" + str(valid_batches), end="")
                    lower_index = j*cfg.batch_size
                    upper_index = min(len(valid_images), ((j+1)*cfg.batch_size))

                    v_imgs, v_labels, v_obj_detection = load_files(
                        valid_images[lower_index:upper_index])

                    v_imgs = (np.array(v_imgs)/127.5)-1

                    v_labels = np.array(v_labels)

                    v_obj_detection = np.array(v_obj_detection)

                    if len(v_labels) == 0:
                        continue

                    if cfg.enable_logging and i == 0:
                        merge = tf.summary.merge_all()
                        summary, _ = sess.run([merge, yolo.loss], feed_dict={
                            yolo.train_bounding_boxes: v_labels,
                            yolo.train_object_recognition: v_obj_detection,
                            yolo.x: v_imgs,
                            yolo.anchors: anchors
                        })

                        train_writer.add_summary(summary, 0)

                    predictions, \
                    loss, lp, ld, lo, lc, \
                    true_pos, false_pos, false_neg, cla = sess.run([
                        yolo.pred_boxes,
                        yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                        yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.pred_classes], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: 0.5
                    })

                    # keep track of true/false positive values to calculate mAP
                    tps = true_pos
                    fps = false_pos
                    fns = false_neg

                    for p in range(9):

                        tp, fp, fn = sess.run([
                            yolo.true_positives, yolo.false_positives, yolo.false_negatives], feed_dict={
                            yolo.train_bounding_boxes: v_labels,
                            yolo.train_object_recognition: v_obj_detection,
                            yolo.x: v_imgs,
                            yolo.anchors: anchors,
                            yolo.iou_threshold: (0.55 + (0.05 * p))
                        })

                        tps += tp
                        fps += fp
                        fns += fn

                    del(v_imgs, v_labels, v_obj_detection, predictions)

                    losses[0] += loss
                    losses[1] += lp
                    losses[2] += ld
                    losses[3] += lo
                    losses[4] += lc

                    #precision
                    losses[5] += true_pos / (true_pos + false_pos) / valid_batches

                    #recall
                    losses[6] += true_pos / (true_pos + false_neg) / valid_batches

                    #mAP
                    losses[7] += (tps / (tps + fps)) / valid_batches

                print(i, "loss:", losses)

                loss_string = str(i) + "," + "Validation"

                for l in range(len(losses)):
                    loss_string = loss_string + "," + str(losses[l])


                with open("training.csv", "a") as file:
                    file.write(loss_string + "\n")

                print(loss_string)

                #learning_r = (cfg.learning_rate_start-cfg.learning_rate_min)*pow(cfg.learning_rate_decay, i) \
                #             + cfg.learning_rate_min

                learning_r = modify_learning_rate(i)
                print("Learning rate:", learning_r)
                yolo.set_training(True)

                losses = [0, 0, 0, 0, 0, 0, 0, 0]

                for j in range(batches):
                    gc.collect()
                    if (cfg.enable_logging):
                        merge = tf.summary.merge_all()

                    print("\rTraining " + str(j) + "/" + str(batches), end="")

                    lower_index = j * cfg.batch_size
                    upper_index = min(len(training_images), (j+1)*cfg.batch_size)
                    imgs, labels, obj_detection = load_files(
                        training_images[lower_index:upper_index])

                    imgs = (np.array(imgs)/127.5)-1

                    labels = np.array(labels)

                    obj_detection = np.array(obj_detection)


                    if len(labels) == 0:
                        continue

                    if (cfg.enable_logging):
                        summary, _, predictions, loss, lp, ld, lo, lc, true_pos, false_pos, false_neg = sess.run([
                            merge, train_step, yolo.pred_boxes,
                            yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                            yolo.true_positives, yolo.false_positives, yolo.false_negatives], feed_dict={
                            yolo.train_bounding_boxes: labels,
                            yolo.train_object_recognition: obj_detection,
                            yolo.x: imgs,
                            yolo.anchors: anchors,
                            learning_rate: learning_r,
                            yolo.iou_threshold: 0.5
                        })

                        tps = true_pos
                        fps = false_pos
                        fns = false_neg

                        losses[0] += loss
                        losses[1] += lp
                        losses[2] += ld
                        losses[3] += lo
                        losses[4] += lc

                        #precision
                        losses[5] += true_pos / (true_pos + false_pos)

                        #recall
                        losses[6] += true_pos / (true_pos + false_neg)

                        #mAP
                        losses[7] += 0#(tps / (tps + fps))


                        train_writer.add_summary(summary, i+1)
                    else:
                        _, predictions, loss, lp, ld, lo, lc, true_pos, false_pos, false_neg = sess.run([
                            train_step, yolo.pred_boxes,
                            yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                            yolo.true_positives, yolo.false_positives, yolo.false_negatives], feed_dict={
                            yolo.train_bounding_boxes: labels,
                            yolo.train_object_recognition: obj_detection,
                            yolo.x: imgs,
                            yolo.anchors: anchors,
                            learning_rate: learning_r,
                            yolo.iou_threshold: 0.5
                        })

                        tps = true_pos
                        fps = false_pos
                        fns = false_neg

                        losses[0] += loss
                        losses[1] += lp
                        losses[2] += ld
                        losses[3] += lo
                        losses[4] += lc

                        #precision
                        losses[5] += true_pos / (true_pos + false_pos)

                        #recall
                        losses[6] += true_pos / (true_pos + false_neg)

                        #mAP
                        losses[7] += 0#(tps / (tps + fps))



                    del(imgs)
                    del(labels)
                    del(obj_detection)


                loss_string = str(i) + "," + "Training"

                for l in range(len(losses)):
                    loss_string = loss_string + "," + str(losses[l])


                with open("training.csv", "a") as file:
                    file.write(loss_string + "\n")

                print(loss_string)


                if i % 10 == 0:
                    save_path = saver.save(sess, str(i) + model_file)

                save_path = saver.save(sess, model_file)


            gc.collect()

            sys.exit()