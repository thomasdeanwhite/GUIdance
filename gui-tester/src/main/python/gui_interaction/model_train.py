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

last_epochs = 0

tf.logging.set_verbosity(tf.logging.INFO)


def modify_learning_rate(epoch):


    ep = epoch

    #return learning rate in accordance to YOLO paper
    if ep == 0:
        return 0.001
    if ep < 10:
        return 0.001+(0.01-0.001)/((10-ep))

    if ep < 75:
        return 0.01

    if ep < 105:
        return 0.001

    return 0.01 / ep


if __name__ == '__main__':

    training_file = cfg.data_dir + "/" + cfg.train_file

    pattern = re.compile(".*\/([0-9]+).*")

    training_images = []

    real_images = []

    with open(training_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])
            training_images.append(l.strip())



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

    print("Found", len(real_images), "real GUI screenshots.")

    #valid_images = random.sample(valid_images, cfg.batch_size)

    with tf.device(cfg.gpu):

        tf.reset_default_graph()

        yolo = Yolo()

        yolo.create_network()

        yolo.set_training(True)

        yolo.create_training()

        global_step = tf.placeholder(tf.int32)
        batches = math.ceil(len(training_images)/cfg.batch_size) if cfg.run_all_batches else 1


        learning_rate = tf.train.exponential_decay(0.1, global_step,
                                                   1, 0.95, staircase=True)
        #learning_rate = tf.placeholder(tf.float64)
        #learning_r = cfg.learning_rate_start
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            yolo.set_update_ops(update_ops)

            train_step = tf.train.AdadeltaOptimizer(learning_rate). \
                minimize(yolo.loss)

        saver = tf.train.Saver()

        model_file = cfg.weights_dir + "/model.ckpt"

        valid_batches = math.ceil(len(valid_images)/cfg.batch_size) if cfg.run_all_batches else 1

        real_batches = math.ceil(len(real_images)/cfg.batch_size) if cfg.run_all_batches else 1

        if (real_batches < 1):
            real_batches = 1

        config = tf.ConfigProto(allow_soft_placement = True)#, log_device_placement=True)
        config.gpu_options.allow_growth = True

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

            if last_epochs == 0:
                with open("training.csv", "w") as file:
                    file.write("epoch,dataset,loss,loss_position,loss_dimension,loss_obj,loss_class,precision,recall,mAP\n")
            else:
                with open("training.csv", "r") as file:
                    print(file.read())


            for i in range(last_epochs, cfg.epochs):
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

                    assert not np.any(np.isnan(v_labels))
                    assert not np.any(np.isnan(v_obj_detection))
                    assert not np.any(np.isnan(v_imgs))

                    iou, iou2, iou3, predictions, \
                    loss, lp, ld, lo, lc, \
                    true_pos, false_pos, false_neg, mAP = sess.run([
                        yolo.loss_layers['pred_boxes_wh'], yolo.loss_layers['dim_loss'], yolo.cell_grid,
                        yolo.pred_boxes,
                        yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                        yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: 0.5,
                        yolo.object_detection_threshold: cfg.object_detection_threshold
                    })

                    # print(predictions[0, 0:3, 0:3, 0, 2:4])
                    # #print(iou[0, 0:3, 0:3, 0, 0:2])
                    # print("---")
                    # print(v_labels[0, 0:3, 0:3, 2:4])
                    # print("---")
                    # #print(iou2[0, 0:3])
                    # print(iou3[0, 0:3, 0:3, 0])
                    #
                    # print("pred:", np.sqrt(predictions[0, 0, 0:3, 0, 2:4]))
                    # print("truth:", np.sqrt(v_labels[0, 0, 0:3, 2:4]))
                    # print("---")
                    #
                    # print(np.square(np.subtract(np.sqrt(predictions[0, 0, 0:3, 0, 2:4]), np.sqrt(v_labels[0, 0, 0:3, 2:4]))))
                    # print(loss)
                    # print(np.min(predictions[..., 2:4]), "-", np.max(predictions[..., 2:4]))
                    # print(np.min(v_labels[..., 2:4]), "-", np.max(v_labels[..., 2:4]))
                    # print(np.min(v_imgs), "-", np.max(v_imgs))
                    # print(np.min(v_obj_detection), "-", np.max(v_obj_detection))

                    # keep track of true/false positive values to calculate mAP
                    tps = true_pos
                    fps = false_pos
                    fns = false_neg

                    del(v_imgs, v_labels, v_obj_detection, predictions)

                    losses[0] += loss
                    losses[1] += lp
                    losses[2] += ld
                    losses[3] += lo
                    losses[4] += lc

                    #precision
                    losses[5] += (true_pos+1) / (true_pos + false_pos+1)

                    #recall
                    losses[6] += (true_pos+1) / (true_pos + false_neg+1)

                    losses[7] += mAP

                for li in range(len(losses)):
                    losses[li] = losses[li] / valid_batches

                loss_string = str(i) + "," + "Validation"

                for l in range(len(losses)):
                    loss_string = loss_string + "," + str(losses[l])


                with open("training.csv", "a") as file:
                    file.write(loss_string + "\n")

                print(loss_string)

                yolo.set_training(False)

                losses = [0, 0, 0, 0, 0, 0, 0, 0]

                for j in range(real_batches):
                    gc.collect()

                    lower_index = j*cfg.batch_size
                    upper_index = min(len(real_images), ((j+1)*cfg.batch_size))

                    v_imgs, v_labels, v_obj_detection = load_files(
                        real_images[lower_index:upper_index])

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
                    true_pos, false_pos, false_neg, mAP = sess.run([
                        yolo.pred_boxes,
                        yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                        yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                        yolo.train_bounding_boxes: v_labels,
                        yolo.train_object_recognition: v_obj_detection,
                        yolo.x: v_imgs,
                        yolo.anchors: anchors,
                        yolo.iou_threshold: 0.5,
                        yolo.object_detection_threshold: cfg.object_detection_threshold
                    })

                    # keep track of true/false positive values to calculate mAP
                    tps = true_pos
                    fps = false_pos
                    fns = false_neg

                    del(v_imgs, v_labels, v_obj_detection, predictions)

                    losses[0] += loss
                    losses[1] += lp
                    losses[2] += ld
                    losses[3] += lo
                    losses[4] += lc

                    #precision
                    losses[5] += (true_pos+1) / (true_pos + false_pos+1)

                    #recall
                    losses[6] += (true_pos+1) / (true_pos + false_neg+1)

                    losses[7] += mAP

                for li in range(len(losses)):
                    losses[li] = losses[li] / real_batches

                loss_string = str(i) + "," + "Real"

                for l in range(len(losses)):
                    loss_string = loss_string + "," + str(losses[l])


                with open("training.csv", "a") as file:
                    file.write(loss_string + "\n")

                print(loss_string)

                #learning_r = (cfg.learning_rate_start-cfg.learning_rate_min)*pow(cfg.learning_rate_decay, i) \
                #             + cfg.learning_rate_min

                learning_r = modify_learning_rate(i)
                #print("Learning rate:", learning_r)
                yolo.set_training(True)

                #print("\rTraining from " + str(batches) + " batches.", end="")

                losses = [0, 0, 0, 0, 0, 0, 0, 0]

                progress_t = 0

                for j in range(batches):

                    if i == 0 and j/batches > progress_t:
                        print("Progress:", round(100*progress_t), "%.")
                        progress_t += 0.1

                    gc.collect()
                    if (cfg.enable_logging):
                        merge = tf.summary.merge_all()



                    lower_index = j * cfg.batch_size
                    upper_index = min(len(training_images), (j+1)*cfg.batch_size)
                    imgs, labels, obj_detection = load_files(
                        training_images[lower_index:upper_index])

                    imgs = (np.array(imgs)/127.5)-1

                    labels = np.array(labels)

                    obj_detection = np.array(obj_detection)


                    if len(labels) == 0:
                        continue

                    assert not np.any(np.isnan(labels))
                    assert not np.any(np.isnan(obj_detection))
                    assert not np.any(np.isnan(imgs))

                    if (cfg.enable_logging):
                        summary, _, predictions, loss, lp, ld, lo, lc, true_pos, false_pos, false_neg, mAP = sess.run([
                            merge, train_step, yolo.pred_boxes,
                            yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                            yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                            yolo.train_bounding_boxes: labels,
                            yolo.train_object_recognition: obj_detection,
                            yolo.x: imgs,
                            yolo.anchors: anchors,
                            # learning_rate: learning_r,
                            global_step: i,
                            yolo.iou_threshold: 0.5,
                            yolo.object_detection_threshold: cfg.object_detection_threshold
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
                        losses[5] += (true_pos+1) / (true_pos + false_pos+1)

                        #recall
                        losses[6] += (true_pos+1) / (true_pos + false_neg+1)

                        #mAP
                        losses[7] += mAP


                        train_writer.add_summary(summary, i+1)
                    else:
                        _, predictions, loss, lp, ld, lo, lc, true_pos, false_pos, false_neg, mAP = sess.run([
                            train_step, yolo.pred_boxes,
                            yolo.loss, yolo.loss_position, yolo.loss_dimension, yolo.loss_obj, yolo.loss_class,
                            yolo.true_positives, yolo.false_positives, yolo.false_negatives, yolo.mAP], feed_dict={
                            yolo.train_bounding_boxes: labels,
                            yolo.train_object_recognition: obj_detection,
                            yolo.x: imgs,
                            yolo.anchors: anchors,
                            #learning_rate: learning_r,
                            global_step: i,
                            yolo.iou_threshold: 0.5,
                            yolo.object_detection_threshold: cfg.object_detection_threshold
                        })

                        tps = true_pos
                        fps = false_pos
                        fns = false_neg

                        if (np.isnan(ld) or np.isinf(ld) or np.any(np.isinf(predictions))):
                            print("INF!")
                            print(np.max(labels))
                            print(np.max(imgs))
                            print("---")

                        losses[0] += loss
                        losses[1] += lp
                        losses[2] += ld
                        losses[3] += lo
                        losses[4] += lc

                        #precision
                        losses[5] += (true_pos+1) / (true_pos + false_pos+1)

                        #recall
                        losses[6] += (true_pos+1) / (true_pos + false_neg+1)

                        #mAP
                        losses[7] += mAP



                    del(imgs)
                    del(labels)
                    del(obj_detection)

                for li in range(len(losses)):
                    losses[li] = losses[li] / batches


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