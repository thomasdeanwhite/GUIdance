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

def load_file(files):
    images = []

    for f in files:
        image = cv2.imread(f, 0)
        img_raw = cv2.imread(f)
        image = cv2.resize(image, (cfg.width, cfg.height))
        image = np.reshape(image, [cfg.width, cfg.height, 1])
        images.append([image, img_raw])

    return images



if __name__ == '__main__':

    with tf.device(cfg.gpu):



        tf.reset_default_graph()
        yolo = Yolo()

        yolo.create_network()
        #yolo.set_training(False)
        #yolo.create_training()

        learning_rate = tf.placeholder(tf.float64)
        learning_r = cfg.learning_rate_start

        saver = tf.train.Saver()

        model_file = os.getcwd() + "/weights/model.ckpt"

        #chkp.print_tensors_in_checkpoint_file(model_file, tensor_name='', all_tensors=True)

        config = tf.ConfigProto(allow_soft_placement = True)

        with tf.Session(config=config) as sess:

            init_op = tf.global_variables_initializer()
            model = sess.run(init_op)
            if os.path.isfile(os.getcwd() + "/weights/checkpoint"):
                saver.restore(sess, model_file)
                print("Restored model")
            yolo.set_training(False)

            anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
            images = load_file(sys.argv[1:])

            #normalise data  between 0 and 1
            imgs = (np.array([row[0] for row in images])/255)

            boxes = sess.run(yolo.output, feed_dict={
                yolo.x: imgs,
                yolo.anchors: anchors,
            })

            proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=True)


            for box in proc_boxes:
                cls = yolo.names[int(box[0])]

                hex = cls.encode('utf-8').hex()[0:6]

                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

                print(cls, box[1], box[2], box[3], box[4], box[5])