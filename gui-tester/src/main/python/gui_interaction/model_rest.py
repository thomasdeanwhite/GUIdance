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
import cherrypy
import json

anchors = None
yolo = None
sess = None

class RestPrediction (object):
    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def predictions(self):
        input = cherrypy.request.json
        img = input["image"]

        result = process_image(img)

        json_img = json.dumps(result)

        return json_img

def neaten_boxes(proc_boxes, img):
    height, width = img.shape[:2]

    # plot boxes
    for box in proc_boxes:

        box[0] = yolo.names[int(box[0])]

        box[1] = max(int(width*box[1]), 0)
        box[2] = max(int(height*box[2]), 0)
        box[3] = int(width*box[3])
        box[4] = int(height*box[4])
    return proc_boxes

def process_image(img_file):

    raw_img = load_raw_image(img_file)

    img = np.array([load_image(img_file)])[0]

    images = np.array([img])

    #normalise data  between 0 and 1
    imgs = np.array(images)/127.5-1

    boxes = sess.run(yolo.output, feed_dict={
        yolo.x: imgs,
        yolo.anchors: anchors,
    })

    height, width = raw_img.shape[:2]

    proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=False).tolist()[0]

    proc_boxes.sort(key=lambda box: -box[5])

    print("Processing Boxes")

    proc_boxes = neaten_boxes(yolo.normalise_boxes(proc_boxes, width, height), raw_img)

    return proc_boxes

if __name__ == '__main__':

    disable_transformation()

    yolo = Yolo()

    yolo.create_network()

    model_file = os.getcwd() + "/" + cfg.weights_dir + "/model.ckpt"

    yolo.prepare()

    config = tf.ConfigProto(device_count = {'GPU': 0})

    with tf.Session(config=config) as ses:

        sess = ses

        if not yolo.init_session(sess, model_file):
            print("Cannot find weights file.", file=sys.stderr)
            sys.exit(-1)

        yolo.set_training(False)

        anchors = np.reshape(np.array(cfg.anchors), [-1, 2])

        cherrypy.quickstart(RestPrediction())
