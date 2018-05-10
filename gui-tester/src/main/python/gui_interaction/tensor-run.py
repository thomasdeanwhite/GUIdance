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

            model_file = os.getcwd() + "/model/model.ckpt"

            valid_batches = math.ceil(len(valid_images)/cfg.batch_size)

            config = tf.ConfigProto(allow_soft_placement = True)

            with tf.Session(config=config) as sess:

                init_op = tf.global_variables_initializer()
                model = sess.run(init_op)
                if os.path.isfile(os.getcwd() + "/model/checkpoint"):
                    saver.restore(sess, model_file)
                    print("Restored model")

                image_length = len(training_images)
                batches = math.ceil(image_length/cfg.batch_size)

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

                            color = tuple(int(hex[i:i+2], 16) for i in (0, 2 ,4))

                            for k in range(int(len(cfg.anchors)/2)):
                                box = cell
                                img = images[image][1]
                                if (box[k+4]>0.6):
                                    box[0] = (0.5+i)*i_offset+box[0]
                                    box[1] = (0.5+j)*j_offset+box[1]
                                    print(image, box[k*5:(k+1)*5])

                                    height, width, channels = img.shape

                                    avg_col = color[0] + color[1] + color[2]

                                    text_col = (255, 255, 255)

                                    if avg_col > 127:
                                        text_col = (0, 0, 0)

                                    cv2.rectangle(img,
                                                  (int(width*(box[0]-box[2]/2)),int(height*(box[1]-box[3]/2))),
                                                  (int(width*((box[0]+box[2]/2))), int(height*(box[1]+box[3]/2))),
                                                  (color[0], color[1], color[2]), int(10*box[4]), 8)
                                    cv2.rectangle(img,
                                                  (int(width*(box[0]-box[2]/2)), int(height*(box[1]-box[3]/2))-int(10*box[4])-15),
                                                  (int(width*(box[0]-box[2]/2)) + len(cls)*7,
                                                   int(height*(box[1]-box[3]/2))),
                                                  (color[0], color[1], color[2]), -1, 8)
                                    cv2.putText(img, cls,
                                                ((int(width*(box[0]-box[2]/2)), int(height*(box[1]-box[3]/2))-int(10*box[4])-2)),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.4, text_col, 1)

                cv2.imshow('image',images[0][1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()