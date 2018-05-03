import config as cfg
from yolo import Yolo
import cv2
import numpy as np
import tensorflow as tf
import sys
import gc
import math

def normalise_point(point, val):
    v = point*val
    return (v - np.round(v))/v

def normalise_label(label):
    return([
        normalise_point(label[0], cfg.grid_shape[0]),
        normalise_point(label[1], cfg.grid_shape[1]),
        label[2],
        label[3],
        label[4]
    ])

def load_files(files):
    label_files = [f.replace("/images/", "/labels/") for f in files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    images = []
    labels = []
    object_detection = []

    for f in files:
        image = cv2.imread(f, 0)
        image = cv2.resize(image, (cfg.width, cfg.height))
        image = np.reshape(image, [cfg.width, cfg.height, 1])
        images.append(image)

    for f in label_files:
        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]
        with open(f, "r") as l:
            obj_detect = [[[0 for i in
                           range(len(yolo.names))] for i in
                          range(cfg.grid_shape[0])]for i in
                range(cfg.grid_shape[1])]
            imglabs = [[[0 for i in
                        range(5)]for i in
                       range(cfg.grid_shape[1])] for i in
                range(cfg.grid_shape[0])]
            for line in l:
                list = line.split(" ")
                x = max(0, min(round(float(list[1])*cfg.grid_shape[0])-1, cfg.grid_shape[0]-1))
                y = max(0, min(round(float(list[2])*cfg.grid_shape[1])-1, cfg.grid_shape[1]-1))
                normalised_label = normalise_label([float(list[1]), float(list[2]), float(list[3]), float(list[4]), 1])
                imglabs[y][x] = normalised_label
                obj_detect[y][x] = [0 for i in range(len(yolo.names))]
                obj_detect[y][x][int(list[0])] = 1

            object_detection.append(obj_detect)
            labels.append(imglabs)
    return images, labels, object_detection



if __name__ == '__main__':
    training_file = cfg.data_dir + "/" + cfg.train_file

    training_images = []

    with open(training_file, "r") as tfile:
        for l in tfile:
            training_images.append(l.strip())

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

        model_file = "model/model.ckpt"

        with tf.Session() as sess:

            init_op = tf.global_variables_initializer()

            print("Initialising Memory Values")
            model = sess.run(init_op)
            print("!Finished Initialising Memory Values!")
            image_length = len(training_images)
            batches = math.ceil(image_length/cfg.batch_size)
            print("Starting training:", image_length, "images in", batches, "batches.")

            anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
            print("anchors", anchors.shape)

            for i in range(cfg.epochs):
                learning_r = max(cfg.learning_rate_min, cfg.learning_rate_start*pow(cfg.learning_rate_decay, i))
                print("Learning rate:", learning_r)
                for j in range(batches):
                    lower_index = j*cfg.batch_size
                    upper_index = min(len(training_images), ((j+1)*cfg.batch_size))
                    imgs, labels, obj_detection = load_files(
                        training_images[lower_index:upper_index])

                    imgs = np.array(imgs)/255

                    labels = np.array(labels)


                    obj_detection = np.array(obj_detection)

                    sess.run(train_step, feed_dict={
                        yolo.train_bounding_boxes: labels,
                        yolo.train_object_recognition: obj_detection,
                        yolo.x: imgs,
                        yolo.anchors: anchors,
                        learning_rate: learning_r
                    })

                    # print("bool:", sess.run(yolo.bool, feed_dict={
                    #     yolo.train_bounding_boxes: labels,
                    #     yolo.train_object_recognition: obj_detection,
                    #     yolo.x: imgs,
                    #     yolo.anchors: anchors
                    # }))

                    # sess.run(train_step, feed_dict={
                    #     yolo.train_bounding_boxes: labels,
                    #     yolo.train_object_recognition: obj_detection,
                    #     yolo.x: imgs,
                    #     yolo.anchors: anchors,
                    #     learning_rate: learning_r
                    # })

                    loss = sess.run(yolo.loss, feed_dict={
                        yolo.train_bounding_boxes: labels,
                        yolo.train_object_recognition: obj_detection,
                        yolo.x: imgs,
                        yolo.anchors: anchors
                    })

                    del(imgs)
                    del(labels)
                    del(obj_detection)

                    print("loss:", loss)

                if i % 10 == 0:
                    save_path = saver.save(sess, str(i) + model_file)
                    print("Model saved in file: %s" % save_path)


            gc.collect()

            sys.exit()