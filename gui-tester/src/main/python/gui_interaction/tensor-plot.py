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

def normalise_point(point, val):
    v = point*val
    return (v - np.round(v))/float(int(v)+1)

def normalise_label(label):
    return([
        normalise_point(max(0, min(1, label[0])), cfg.grid_shape[0]),
        normalise_point(max(0, min(1, label[1])), cfg.grid_shape[1]),
        max(0, min(1, label[2])),
        max(0, min(1, label[3])),
        label[4]
    ])

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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            yolo = Yolo()

            yolo.create_network()

            yolo.set_training(True)
            yolo.set_update_ops(update_ops)

            yolo.create_training()

            learning_rate = tf.placeholder(tf.float64)
            learning_r = cfg.learning_rate_start

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
                images = load_file(sys.argv[1:])

                imgs = (np.array([row[0] for row in images])/127.5)-1

                boxes = sess.run(yolo.output, feed_dict={
                    yolo.x: imgs,
                    yolo.anchors: anchors,
                })

                i_offset = 1/cfg.grid_shape[0]
                j_offset = 1/cfg.grid_shape[1]

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
                                    print(plot_box)
                                    plot_box[0] = (0.5+i)*i_offset+plot_box[0]
                                    plot_box[1] = (0.5+j)*j_offset+plot_box[1]
                            box = plot_box
                            if (box[4]>cfg.object_detection_threshold):
                                img = images[image][1]
                                print(cls, box)

                                height, width, channels = img.shape

                                avg_col = color[0] + color[1] + color[2]

                                text_col = (255, 255, 255)

                                if avg_col > 127:
                                    text_col = (0, 0, 0)

                                x1 = max(int(width*(box[0]-box[2]/2)), 0)
                                y1 = max(int(height*(box[1]-box[3]/2)), 0)
                                x2 = int(width*((box[0]+box[2]/2)))
                                y2 = int(height*(box[1]+box[3]/2))

                                cv2.rectangle(img, (x1, y1),
                                              (x2, y2),
                                              (color[0], color[1], color[2]), int(10*box[4]), 8)

                                cv2.rectangle(img,
                                              (x1, y1-int(10*box[4])-15),
                                              (x1 + len(cls)*7, y1),
                                              (color[0], color[1], color[2]), -1, 8)

                                cv2.putText(img, cls,
                                            (x1, y1-int(10*box[4])-2),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4, text_col, 1)

                cv2.imshow('image',images[0][1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()