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
        image = np.int16(cv2.imread(f, 0))
        img_raw = cv2.imread(f)

        # if random.random() < cfg.brightness_probability:
        #     brightness = int(random.random()*cfg.brightness_var*2)-cfg.brightness_var
        #     image = np.maximum(0, np.minimum(255, np.add(image, brightness)))
        #
        # if random.random() < cfg.contrast_probability:
        #     contrast = (random.random() * cfg.contrast_var * 2) - cfg.contrast_var
        #
        #     contrast_diff = (image - np.mean(image)) * contrast
        #     image = np.maximum(0, np.minimum(255, np.add(image, contrast_diff)))
        #
        # if random.random() < cfg.invert_probability:
        #     image = 255 - image
        #
        image = np.uint8(cv2.resize(image, (cfg.width, cfg.height)))
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

        model_file = os.getcwd() + "/" + cfg.weights_dir + "/model.ckpt"

        #chkp.print_tensors_in_checkpoint_file(model_file, tensor_name='', all_tensors=True)

        config = tf.ConfigProto(allow_soft_placement = True)

        with tf.Session(config=config) as sess:

            init_op = tf.global_variables_initializer()
            model = sess.run(init_op)
            if os.path.isfile(os.getcwd() + "/" + cfg.weights_dir + "/checkpoint"):
                saver.restore(sess, model_file)
                print("Restored model")
            yolo.set_training(False)

            anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
            images = load_file(sys.argv[1:])

            #normalise data  between 0 and 1
            imgs = (np.array([row[0] for row in images])/127.5)-1

            boxes = sess.run(yolo.output, feed_dict={
                yolo.x: imgs,
                yolo.anchors: anchors,
            })

            proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=True).tolist()


            img = images[0][1]

            for box in proc_boxes:
                height, width, channels = img.shape

                cls = yolo.names[int(box[0])]

                hex = cls.encode('utf-8').hex()[0:6]

                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

                if (box[5]>cfg.object_detection_threshold):
                    print(box)

                    x1 = max(int(width*(box[1]-box[3]/2)), 0)
                    y1 = max(int(height*(box[2]-box[4]/2)), 0)
                    x2 = int(width*((box[1]+box[3]/2)))
                    y2 = int(height*(box[2]+box[4]/2))

                    cv2.rectangle(img, (x1, y1),
                                  (x2, y2),
                                  (color[0], color[1], color[2]), 3+int(7*box[4]), 8)

            for box in proc_boxes:
                cls = yolo.names[int(box[0])]

                hex = cls.encode('utf-8').hex()[0:6]

                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

                if (box[5]>cfg.object_detection_threshold):
                    height, width, channels = img.shape

                    avg_col = color[0] + color[1] + color[2]

                    text_col = (255, 255, 255)

                    if avg_col > 127:
                        text_col = (0, 0, 0)

                    x1 = max(int(width*(box[1]-box[3]/2)), 0)
                    y1 = max(int(height*(box[2]-box[4]/2)), 0)
                    x2 = int(width*((box[1]+box[3]/2)))
                    y2 = int(height*(box[2]+box[4]/2))

                    cv2.rectangle(img,
                                  (x1, y1-int(10*box[4])-15),
                                  (x1 + (5 + len(cls))*7, y1),
                                  (color[0], color[1], color[2]), -1, 8)

                    cv2.putText(img, cls + str(round(box[5]*100)),
                                (x1, y1-int(10*box[4])-2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, text_col, 1)

            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
