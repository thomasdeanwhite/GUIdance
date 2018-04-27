import config as cfg
from yolo import Yolo
import cv2
import numpy as np
import tensorflow as tf
import sys
import gc

def load_files(files):
    label_files = [f.replace("/images/", "/labels/") for f in files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    images = []
    labels = []
    object_detection = []

    for f in files:
        image = cv2.imread(f, 0)
        image = cv2.resize(image, (cfg.width, cfg.height))
        image = np.reshape(image, [1, cfg.width, cfg.height, 1])
        images.append(image)

    for f in label_files:
        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]
        imglabs = []
        with open(f, "r") as l:
            obj_detect = [[[0]]*cfg.grid_shape[0]]*cfg.grid_shape[1]
            for line in l:

                list = line.split(" ")
                imglabs.append([float(list[1]), float(list[2]), float(list[3]), float(list[4])])
                x = int(float(list[1])*cfg.grid_shape[0])
                y = int(float(list[2])*cfg.grid_shape[1])
                obj_detect[y][x] = [int(list[0])]

            object_detection.append(obj_detect)
            labels.append(imglabs)
    return images, labels, object_detection



if __name__ == '__main__':
    training_file = cfg.data_dir + "/" + cfg.test_file

    training_images = []

    with open(training_file, "r") as tfile:
        for l in tfile:
            training_images.append(l.strip())


    yolo = Yolo()

    yolo.create_network()

    yolo.create_training()

    train_step = tf.train.MomentumOptimizer (cfg.learning_rate_start,
                                             cfg.momentum).minimize(yolo.loss)

    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()

        print("Initialising Memory Values")
        model = sess.run(init_op)
        print("!Finished Initialising Memory Values!")

        for i in range(cfg.epochs):
            for j in range(int(len(training_file)/cfg.batch_size)+1):

                imgs, labels, obj_detection = load_files(training_images[(j*cfg.batch_size):((j+1)*cfg.batch_size)])

                imgs = np.array(imgs[0])
                labels = np.array(labels)
                obj_detection = np.array(obj_detection)


                # print("diou:", sess.run(yolo.d_best_iou, feed_dict={
                #     yolo.train_bounding_boxes: labels,
                #     yolo.train_object_recognition: obj_detection,
                #     yolo.x: imgs
                # }))
                #
                # print("best_iou:", sess.run(yolo.best_iou, feed_dict={
                #     yolo.train_bounding_boxes: labels,
                #     yolo.train_object_recognition: obj_detection,
                #     yolo.x: imgs
                # }))
                #
                # bool = sess.run(tf.shape(yolo.bool), feed_dict={
                #     yolo.train_bounding_boxes: labels,
                #     yolo.train_object_recognition: obj_detection,
                #     yolo.x: imgs
                # })
                #
                # print("bool:", bool)
                #
                # bool = sess.run(yolo.bool, feed_dict={
                #     yolo.train_bounding_boxes: labels,
                #     yolo.train_object_recognition: obj_detection,
                #     yolo.x: imgs
                # })
                #
                # print("bool:", bool)




                loss = sess.run(train_step, feed_dict={
                    yolo.train_bounding_boxes: labels,
                    yolo.train_object_recognition: obj_detection,
                    yolo.x: imgs
                })

                print("loss:", loss)

                print()

                gc.collect()

                sys.exit()