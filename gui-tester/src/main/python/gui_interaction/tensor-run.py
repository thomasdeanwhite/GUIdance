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
import pyautogui
from operator import itemgetter
import time

if __name__ == '__main__':

    program_name = "Firefox"

    #info = os.system("xwininfo -name \"" + program_name + "\" | grep Corners")

    #info = info.trim()

    #corners = info.split("[+-]*")

    #print(corners)


    with tf.device(cfg.gpu):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            yolo = Yolo()

            yolo.create_network()

            yolo.set_training(True)
            yolo.set_update_ops(update_ops)

            yolo.create_training()

            learning_rate = tf.placeholder(tf.float64)
            learning_r = cfg.learning_rate_start

            train_step = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4). \
                minimize(yolo.loss)

            saver = tf.train.Saver()

            model_file = os.getcwd() + "/backup_model/model.ckpt"

            config = tf.ConfigProto(allow_soft_placement = True)

            with tf.Session(config=config) as sess:

                init_op = tf.global_variables_initializer()
                model = sess.run(init_op)
                if os.path.isfile(os.getcwd() + "/backup_model/checkpoint"):
                    saver.restore(sess, model_file)
                    print("Restored model")
                yolo.set_training(False)

                anchors = np.reshape(np.array(cfg.anchors), [-1, 2])

                start_time = time.time()

                while (time.time() - start_time < 30):
                    os.system("gnome-screenshot --file=/tmp/current_screen.png")

                    image = cv2.imread("/tmp/current_screen.png", 0)
                    image = cv2.resize(image, (cfg.width, cfg.height))

                    images = np.reshape(image, [1, cfg.width, cfg.height, 1])

                    #imgs = (np.array([row[0] for row in images])/127.5)-1

                    boxes = sess.run(yolo.output, feed_dict={
                        yolo.x: images,
                        yolo.anchors: anchors,
                    })

                    i_offset = 1/cfg.grid_shape[0]
                    j_offset = 1/cfg.grid_shape[1]

                    action = 0

                    proc_boxes = []

                    for image in range(boxes.shape[0]):
                        for i in range(cfg.grid_shape[0]):
                            for j in range(cfg.grid_shape[1]):
                                cell = boxes[image][j][i]
                                classes = cell[int((len(cfg.anchors)/2)*5):]
                                amax = np.argmax(classes)
                                cls = yolo.names[amax]

                                hex = cls.encode('utf-8').hex()[0:6]

                                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

                                plot_box = [0, 0, 0, 0, 0]

                                for k in range(int(len(cfg.anchors)/2)):
                                    box = cell[k*5:(k+1)*5]
                                    if (box[4]>cfg.object_detection_threshold and box[4]>plot_box[4]):
                                        #plot_box = box[k:k+5]
                                        plot_box = box
                                        plot_box[0] = (0.5+i)*i_offset+plot_box[0]
                                        plot_box[1] = (0.5+j)*j_offset+plot_box[1]
                                box = plot_box
                                proc_boxes.append(box)

                    highest_conf = proc_boxes[0][4]
                    best_box = proc_boxes[0]
                    for b in proc_boxes:
                        if (b[4] > highest_conf):
                            highest_conf = b[4]
                            best_box = b
                    x = b[0] * 1920
                    y = b[1] * 1080

                    pyautogui.click(x, y)
                    print("Clicking", "(", x, y, ")")


