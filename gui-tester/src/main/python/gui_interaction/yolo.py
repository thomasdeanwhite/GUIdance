import numpy as np
import tensorflow as tf
import config as cfg

class Yolo:

    x = None
    network = None
    filter = None
    names = []
    loss = None
    train_object_recognition = None
    train_bounding_boxes = None
    input_var = None
    best_iou = None
    bool = None
    d_best_iou = None
    variables = []

    def __init__(self):
        with open(cfg.data_dir + "/" + cfg.names_file, "r") as f:
            for row in f:
                name = row.strip()
                if len(name) > 0:
                    self.names.append(name)

        print(self.names)
        return

    def bbox_overlap_iou(self, bboxes1, bboxes2):
        print(bboxes1.shape, bboxes2.shape)

        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        yI1 = tf.maximum(y11, tf.transpose(y21))

        xI2 = tf.minimum(x12, tf.transpose(x22))
        yI2 = tf.minimum(y12, tf.transpose(y22))

        inter_area = tf.maximum((xI2 - xI1 + 1), 0) * tf.maximum((yI2 - yI1 + 1), 0)

        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

        ret_value = inter_area / union

        return ret_value

    def create_filter(self, size, name):

        var = tf.Variable(tf.truncated_normal(size, stddev=0.02), name=name)
        self.variables.append(var)
        return var
        #
        # return tf.get_variable(name, size, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
        #                    dtype=tf.float32)

    def leaky_relu(self, layer):
        return tf.nn.leaky_relu(
            tf.nn.batch_normalization(layer, 0, 1, None, None, 0.0001),
            0.1)

    def create_network(self):

        anchors_size = len(cfg.anchors)/2
        classes = len(self.names)

        height = int(cfg.height/cfg.grid_shape[1]*anchors_size)
        width = int(cfg.width/cfg.grid_shape[0]*anchors_size)

        self.x = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1], "input")

        #self.input_var = tf.Variable(self.x)

        #self.object_recognition = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1], "obj-rec")
        self.network = self.leaky_relu(tf.nn.conv2d(self.x,
                                    self.create_filter([3,
                                                        3,
                                                        1,
                                                        32], "f1"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)
        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        32,
                                                        64], "f2"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)


        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        64,
                                                        128], "f3"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        128,
                                                        64], "f4"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)


        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        64,
                                                        128], "f5"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)

        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        128,
                                                        256], "f6"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)


        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        256,
                                                        128], "f7"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)

        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        128,
                                                        256], "f8"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)

        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        256,
                                                        512], "f9"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        512,
                                                        256], "f10"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        256,
                                                        512], "f11"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        512,
                                                        256], "f12"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        256,
                                                        512], "f13"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))

        reorg = tf.extract_image_patches(self.network, [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")

        self.network = tf.nn.max_pool(self.network, [1, 1, 1, 1], [1, 2, 2, 1], padding="SAME")
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        512,
                                                        1024], "f14"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        1024,
                                                        512], "f15"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        512,
                                                        1024], "f16"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        1024,
                                                        512], "f17"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        512,
                                                        1024], "f18"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        1024,
                                                        1024], "f19"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        1024,
                                                        1024], "f20"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)

        print("Combining ", self.network.shape, "with reorg:", reorg.shape)
        self.network = tf.concat([self.network, reorg], axis=-1)

        print(self.network.shape)

        self.network = self.leaky_relu(tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        3072,
                                                        1024], "f21"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu))
        print(self.network.shape)


        self.network = tf.nn.conv2d(self.network,
                                    self.create_filter([3,
                                                        3,
                                                        1024,
                                                        int(anchors_size*5 + classes)], "f22"),
                                    [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=cfg.cudnn_on_gpu)
        print(self.network.shape)


    def create_training(self):

        classes = len(self.names)
        print("classes:", classes)
        anchors = int(len(cfg.anchors)/2)
        print("anchors:", anchors)

        self.train_object_recognition = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 1], "train_obj_rec")
        self.train_bounding_boxes = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 4], "train_bb")

        #truth = tf.reshape(self.train_bounding_boxes, [-1, 13*13, 4])

        predictions = tf.reshape(self.network, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], int(anchors*5 + classes)])
        print("pred:", predictions.shape)

        raw_boxes = tf.slice(predictions, [0,0,0], [-1,-1,(anchors*5)])
        print("raw:", raw_boxes.shape)

        pred_boxes_c = tf.reshape(raw_boxes, [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 5, 1])
        print("p_boxes_c:", pred_boxes_c.shape)


        pred_confidence = tf.reshape(
            pred_boxes_c[:, :, 4, :],
            [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 1]
        )

        print("conf:", pred_confidence.shape)

        pred_classes = predictions[:,:, anchors*5:anchors*5+classes]

        print("p_classes:", pred_classes.shape)



        pred_boxes = tf.reshape(pred_boxes_c[:, :, 0:4, :],
                                 [-1, cfg.grid_shape[0], cfg.grid_shape[1], anchors, 4]
        )

        pred_boxes_xy = pred_boxes[:, :, 0:2]
        pred_boxes_wh = tf.square(pred_boxes[:, :, 2:4])

        pred_boxes = tf.concat([pred_boxes_xy, pred_boxes_wh], axis=-1)

        print("p_boxes:", pred_boxes.shape)
        print("truth:", truth.shape)


        wl_start = tf.constant(1)

        c = lambda i, x, y, r: i < tf.shape(x)[0]
        b = lambda i, x, y, r: (tf.add(i, 1), x, y, tf.concat([r, tf.expand_dims(self.bbox_overlap_iou(x[i], y[i]), 0)], axis=0))
        i, x, y, overlap_op = tf.while_loop(c, b, (wl_start, truth, pred_boxes,
                                             tf.expand_dims(self.bbox_overlap_iou(truth[0], pred_boxes[0]), 0)),
                                            shape_invariants=(wl_start.get_shape(),
                                                              truth.get_shape(),
                                                              pred_boxes.get_shape(),
                                                              tf.TensorShape([None, None, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors])))


        #overlap_op = tf.reshape(overlap_op, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], anchors])

        print("overlap:", overlap_op.shape)

        overlap_op = tf.where(tf.is_finite(overlap_op), overlap_op, tf.zeros_like(overlap_op))

        v, indices = tf.nn.top_k(overlap_op)



        #v = tf.tile(v, [1, 1, tf.shape(overlap_op)[2]])

        #indices = tf.reshape(indices, [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 1])

        print("v:", v.shape)
        print("indices:", indices.shape)

        #bool_zeros = tf.zeros_like(overlap_op, dtype=tf.bool)

        #bool_index = tf.scatter_update(bool_zeros, indices, True)

        #bool = tf.cast(bool_index, dtype=tf.bool)

        bool = overlap_op >= v

        #bool = tf.reshape(bool, [tf.shape(bool)[0], tf.shape(bool)[1], 1, tf.shape(bool)[2]])

        print("bool:", bool.shape)

        boxes_replicate = tf.expand_dims(pred_boxes, 1)

        print("boxes_rep:", boxes_replicate.shape)

        wlr_start = tf.constant(1)

        i, x, y, boxes_replicate = tf.while_loop(
            lambda i, x, y, b : i < tf.shape(x)[0],
            lambda i, x, y, b: (tf.add(i, 1), x, y,
                                tf.concat([b,
            tf.reshape(tf.tile(y, [1, tf.shape(x)[0], 1]), [1, -1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 4])], 0)),
            (wlr_start, bool, pred_boxes,
            tf.reshape(tf.tile(pred_boxes[0], [tf.shape(bool[0])[0], 1]), [1, -1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 4])),
            shape_invariants=(wlr_start.get_shape(),
                              bool.get_shape(),
                              pred_boxes.get_shape(),
                              tf.TensorShape([None, None, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 4])))

        #boxes_replicate = tf.map_fn(lambda x : tf.reshape(tf.tile(pred_boxes, [1, tf.shape(x)[0], 1]), [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 4]), bool)

        # boxes_replicate = tf.reshape(boxes_replicate, [tf.shape(boxes_replicate)[0], tf.shape(boxes_replicate)[1],
        #                                                tf.shape(boxes_replicate)[2], 1, tf.shape(boxes_replicate)[3]])
        print("boxes_rep:", boxes_replicate.shape)

        #best_iou = tf.map_fn(lambda x: tf.boolean_mask(box es_replicate, x, name="gather_top_iou"), bool)

        wlb_start = tf.constant(1)

        #bool_test = tf.boolean_mask(boxes_replicate[0], bool[0], axis=-1)




        #default_best_iou = tf.boolean_mask(boxes_replicate, bool, axis=0)

        default_best_iou = tf.expand_dims(tf.gather_nd(pred_boxes[0], indices[0]), 0)

        #default_best_iou = tf.where(overlap_op >= v, boxes_replicate, tf.zeros_like(boxes_replicate))

        print("d_best_iou:", default_best_iou.shape)

        i, x, y, best_iou = tf.while_loop(
            lambda i, x, y, b : i < tf.shape(x)[0],
            lambda i, x, y, b: (tf.add(i, 1), x, y,
                    tf.concat([b, tf.expand_dims(tf.gather_nd(y[i], x[i]), 0)], 0)),
                    (wlb_start, indices, pred_boxes, default_best_iou),
                    shape_invariants=(wlb_start.get_shape(),
                                                indices.get_shape(),
                                                pred_boxes.get_shape(),
                                                tf.TensorShape([None, None, 4])))

        #best_iou = best_iou[:, :, 1]

        print("best_iou:", best_iou.shape)

        self.best_iou = best_iou
        #self.d_best_iou = tf.shape(default_best_iou)
        self.bool = tf.reduce_max(overlap_op)

        loss = tf.reduce_sum(tf.square(truth - best_iou)) * cfg.iou_weight

        # object_recognition = tf.nn.relu(tf.ceil(pred_confidence - cfg.object_detection_threshold)) \
        #                      - tf.minimum(self.train_object_recognition, tf.zeros_like(self.train_object_recognition))
        #
        # loss = loss + (tf.reduce_sum(object_recognition)*cfg.obj_weight)
        #
        # class_onehot = tf.one_hot(tf.cast(self.train_object_recognition, dtype=tf.int32), classes)
        #
        # class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pred_classes, logits=class_onehot)
        #
        # loss = loss + (tf.reduce_sum(class_cross_entropy) * cfg.class_weight)

        self.loss = loss


    def get_network(self):
        return self.network

