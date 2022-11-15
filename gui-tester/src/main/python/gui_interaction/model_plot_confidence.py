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
from data_loader import load_image, load_raw_image, disable_transformation, convert_coords

from model_plot import plot_boxes

if __name__ == '__main__':

    disable_transformation()

    yolo = Yolo()

    yolo.create_network()

    model_file = os.getcwd() + "/" + cfg.weights_dir + "/model.ckpt"

    yolo.prepare()

    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session(config=config) as sess:

        if not yolo.init_session(sess, model_file):
            print("Cannot find weights file.", file=sys.stderr)
            sys.exit(-1)

        yolo.set_training(False)

        anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
        images = np.array([load_image(sys.argv[1])])

        img = images[0]

        #normalise data  between 0 and 1
        imgs = np.array(images)/127.5-1

        boxes = sess.run(yolo.output, feed_dict={
            yolo.x: imgs,
            yolo.anchors: anchors,
        })

        raw_img = load_raw_image(sys.argv[1])

        proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=True).tolist()[0]

        proc_boxes.sort(key=lambda box: -box[5])

        height, width = raw_img.shape[:2]

        proc_boxes = yolo.normalise_boxes(proc_boxes, width, height)

        print("Plotting Figures")
        for c in range(0,11):

            img = np.copy(raw_img)

            p_boxes = proc_boxes[:]

            plot_boxes(p_boxes, img, c/10, yolo)
