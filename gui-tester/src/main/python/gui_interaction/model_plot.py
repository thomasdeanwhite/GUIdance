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

def plot_boxes(proc_boxes, raw_img, threshold, yolo):
    cfg.object_detection_threshold = threshold

    img = np.copy(raw_img)

    proc_boxes = yolo.prune_boxes(proc_boxes[:])


    #proc_boxes = yolo.trim_overlapping_boxes(proc_boxes)

    yolo.plot_boxes(proc_boxes, img)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

        proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=False).tolist()[0]

        raw_img = load_raw_image(sys.argv[1])

        proc_boxes.sort(key=lambda box: -box[5])

        height, width = raw_img.shape[:2]

        print("Processing Boxes")

        proc_boxes = yolo.normalise_boxes(proc_boxes, width, height)

        plot_boxes(proc_boxes, img, cfg.object_detection_threshold, yolo)

