import numpy as np
import tensorflow as tf
import config as cfg

class Yolo:

    x = None
    network = None
    filter = None

    def __init__(self):
        return

    def create_filter(self, size, name):

        return tf.get_variable(name, size, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                               dtype=tf.float32)

    def create_network(self):

        anchors_size = len(cfg.anchors)/2

        height = cfg.height/cfg.grid_shape[1]*anchors_size
        width = cfg.width/cfg.grid_shape[0]*anchors_size

        self.x = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1])
        self.network = tf.nn.conv2d(self.x,
                                    self.create_filter([height,
                                                        width,
                                                        1,
                                                        32], "f1"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)
        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)

        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([height/2,
                                                        width/2,
                                                        32,
                                                        64], "f2"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)

        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)


        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([height/4,
                                                        width/4,
                                                        64,
                                                        128], "f3"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)



    def get_network(self):
        return self.network

