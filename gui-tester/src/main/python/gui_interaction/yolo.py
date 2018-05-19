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
    anchors = None
    is_training = False
    update_ops = None
    loss_position = None
    loss_dimension = None
    loss_obj = None
    loss_noobj = None
    loss_class = None
    epsilon = 0.0001
    output = None
    pred_boxes = None
    pred_classes = None
    loss_layers = {}


    def __init__(self):
        with open(cfg.data_dir + "/" + cfg.names_file, "r") as f:
            for row in f:
                name = row.strip()
                if len(name) > 0:
                    self.names.append(name)

        print(self.names)
        return

    def set_training(self, training):
        self.is_training = training

    def set_update_ops(self, update_ops):
        self.update_ops = update_ops

    def bbox_overlap_iou(self, bboxes1, bboxes2):
        print(bboxes1.shape, bboxes2.shape)

        # x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        # x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)
        x11 = bboxes1[:,:,:,:,0]
        y11 = bboxes1[:,:,:,:,1]
        x12 = bboxes1[:,:,:,:,2]
        y12 = bboxes1[:,:,:,:,3]


        x21 = bboxes2[:,:,:,:,0]
        y21 = bboxes2[:,:,:,:,1]
        x22 = bboxes2[:,:,:,:,2]
        y22 = bboxes2[:,:,:,:,3]

        xI1 = tf.maximum(x11, x21)
        yI1 = tf.maximum(y11, y21)

        xI2 = tf.minimum(x12, x22)
        yI2 = tf.minimum(y12, y22)

        inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI2 - yI1), 0)

        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)

        union = (bboxes1_area + bboxes2_area) - inter_area

        ret_value = inter_area / union

        ret_value = tf.where(tf.logical_or(
            tf.is_inf(ret_value), tf.is_nan(ret_value)), tf.zeros_like(ret_value), ret_value)

        return ret_value

    def create_filter(self, size, name):

        var = tf.Variable(tf.random_normal(size, stddev=cfg.var_sd), name=name)
        self.variables.append(var)
        return var
        #
        # return tf.get_variable(name, size, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
        #                    dtype=tf.float32)

    def leaky_relu(self, layer):
        x = tf.layers.batch_normalization(layer, training=self.is_training)
        return tf.maximum(x, 0.1 * x)

    def create_network(self):

        anchors_size = int(len(cfg.anchors)/2)
        classes = len(self.names)

        height = int(cfg.height/cfg.grid_shape[1]*anchors_size)
        width = int(cfg.width/cfg.grid_shape[0]*anchors_size)

        self.x = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1], "input")

        self.anchors = tf.placeholder(tf.float32, [anchors_size, 2], "anchors")

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

        predictions = tf.reshape(self.network, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], int(anchors_size*5 + classes)])

        raw_boxes = tf.slice(predictions, [0,0,0], [-1,-1,(anchors_size*5)])

        pred_boxes_c = tf.reshape(raw_boxes, [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors_size, 5, 1])


        pred_boxes = tf.reshape(pred_boxes_c[:, :, 0:5, :],
                                [-1, cfg.grid_shape[0], cfg.grid_shape[1], anchors_size, 5]
                                )

        pred_boxes_xy = (pred_boxes[:, :, :, :, 0:2])
        pred_boxes_wh = tf.nn.relu(pred_boxes[:, :, :, :, 2:4])
        anchors_weight = tf.tile(
            tf.reshape(self.anchors, [1, 1, 1, anchors_size, 2]),
            [tf.shape(pred_boxes)[0], cfg.grid_shape[0], cfg.grid_shape[1],
             1, 1])

        pred_boxes_wh = tf.square(tf.multiply(pred_boxes_wh, anchors_weight))

        confidence = (tf.reshape(pred_boxes[:, :, :, :, 4],
                                 [-1, cfg.grid_shape[0], cfg.grid_shape[1], anchors_size, 1]))

        pred_boxes = tf.concat([pred_boxes_xy, pred_boxes_wh], axis=-1)
        pred_boxes = tf.concat([pred_boxes, confidence], axis=-1)

        self.pred_boxes = pred_boxes

        pred_classes = (tf.reshape(
            predictions[:,:, anchors_size*5:anchors_size*5+classes],
            [-1, cfg.grid_shape[0], cfg.grid_shape[1], classes]))

        self.pred_classes = pred_classes

        self.output = tf.concat([tf.reshape(pred_boxes,
                                            [-1, cfg.grid_shape[0], cfg.grid_shape[1], anchors_size*5]
                                 ),
                                 pred_classes], axis=-1)
        print("final network layer:", self.output.shape)


    def create_training(self):

        classes = len(self.names)
        print("classes:", classes)
        anchors = int(len(cfg.anchors)/2)
        print("anchors:", anchors)

        self.train_object_recognition = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], classes], "train_obj_rec")
        self.train_bounding_boxes = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 5], "train_bb")

        truth = tf.reshape(self.train_bounding_boxes, [-1, 13, 13, 1, 5])

        pred_boxes = self.pred_boxes
        print("pred:", pred_boxes.shape)

        pred_confidence = tf.reshape(
            pred_boxes[:, :, 4],
            [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 1]
        )

        print("conf:", pred_confidence.shape)

        pred_classes = self.pred_classes

        print("p_classes:", pred_classes.shape)



        pred_boxes_xy = (pred_boxes[:, :, :, :, 0:2])

        epsilon = tf.constant(self.epsilon)

        pred_boxes_wh = pred_boxes[:, :, :, :, 2:4] + epsilon

        self.loss_layers['pred_boxes_xy'] = pred_boxes_xy
        self.loss_layers['pred_boxes_wh'] = pred_boxes_wh

        truth_boxes_xy = truth[:, :, :, :, 0:2]
        truth_boxes_wh = truth[:, :, :, :, 2:4] + epsilon

        self.loss_layers['truth_boxes_xy'] = truth_boxes_xy
        self.loss_layers['truth_boxes_wh'] = truth_boxes_wh

        pred_wh_half = pred_boxes_wh/2
        pred_min = pred_boxes_xy - pred_wh_half
        pred_max = pred_boxes_xy + pred_wh_half

        true_wh_half = truth_boxes_wh / 2
        true_min = truth_boxes_xy - true_wh_half
        true_max = truth_boxes_xy + true_wh_half

        intersect_mins  = tf.maximum(pred_min,  true_min)
        intersect_maxes = tf.minimum(pred_max, true_max)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = truth_boxes_wh[..., 0] * truth_boxes_wh[..., 1]
        pred_areas = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou = tf.truediv(intersect_areas, union_areas)


        print("iou", iou.shape)


        shaped_iou = tf.reshape(iou, [-1, cfg.grid_shape[0]*cfg.grid_shape[1], anchors])

        indices = tf.reshape(
            tf.argmax(shaped_iou, axis=-1),
            [-1, cfg.grid_shape[0]*cfg.grid_shape[1], 1])

        print("indices", indices.shape)

        shaped_boxes = tf.reshape(pred_boxes, [-1, cfg.grid_shape[0]*cfg.grid_shape[1], anchors, 5])

        print("shaped_boxes", shaped_boxes.shape)

        zero_axis_indices = \
            tf.tile(tf.reshape(tf.range(0, tf.shape(shaped_boxes)[0]), [tf.shape(shaped_boxes)[0], 1, 1]),
                    [1, cfg.grid_shape[0]*cfg.grid_shape[1], 1])

        one_axis_indices = \
            tf.tile(tf.reshape(tf.range(0, tf.shape(shaped_boxes)[1]), [1, cfg.grid_shape[0]*cfg.grid_shape[1], 1]),
                    [tf.shape(shaped_boxes)[0], 1, 1])

        # new_indices = tf.map_fn(lambda x: tf.concat([
        #     zero_axis_indices,
        #     one_axis_indices,
        #     x
        #
        # ], axis=-1), tf.cast(indices, tf.int32))

        new_indices = tf.concat([
            zero_axis_indices,
            one_axis_indices,
            tf.cast(indices, tf.int32)

        ], axis=-1)



        new_indices = tf.reshape(new_indices, [-1, 169, 3])

        self.loss_layers['new_indices'] = new_indices

        iou_reshaped = tf.reshape(iou, [-1, 169, 5, 1])

        print("reshaped_iou", iou_reshaped.shape)

        print("new_indices", new_indices.shape)

        top_boxes = tf.gather_nd(shaped_boxes, new_indices)

        self.loss_layers['top_boxes'] = new_indices

        print("top_boxes", top_boxes.shape)

        top_iou = tf.gather_nd(iou_reshaped, new_indices)

        self.indices = new_indices

        print("top_iou", top_iou.shape)

        top_iou = tf.reshape(top_iou, [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1, 1])

        matching_boxes = tf.reshape(top_boxes, [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1, 5])

        print("matching_boxes", matching_boxes.shape)
        print("truth_boxes", truth.shape)

        self.best_iou = matching_boxes

        self.loss_layers['top_iou'] = top_iou
        self.loss_layers['best_iou'] = matching_boxes

        obj = tf.cast(tf.equal(truth[:,:,:,:,4], 1), tf.float32)

        self.loss_layers['obj'] = obj

        noobj = tf.equal(truth[:,:,:,:,4], 0)

        self.loss_layers['noobj'] = noobj

        obj_xy = tf.reshape(tf.tile(obj,[ 1, 1, 1, 2]),
                            [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1, 2])

        self.loss_layers['obj_xy'] = obj_xy

        iou_losses_xy = tf.square(tf.subtract(truth[:,:,:,:,0:2],
                                                         matching_boxes[:,:,:,:,0:2]))

        iou_losses_xy = obj_xy * iou_losses_xy

        self.loss_layers['iou_losses_xy'] = iou_losses_xy

        iou_losses_wh = tf.square(tf.subtract(tf.sqrt(truth[:,:,:,:,2:4]),
                                       tf.sqrt(matching_boxes[:,:,:,:,2:4])))

        iou_losses_wh = obj_xy * iou_losses_wh

        self.loss_layers['iou_losses_wh'] = iou_losses_wh

        self.loss_position = tf.reduce_sum(iou_losses_xy) *  cfg.coord_weight
        self.loss_dimension = tf.reduce_sum(iou_losses_wh) * cfg.coord_weight

        #pred_conf = tf.multiply(top_iou[:,:,:,:,0], truth[:,:,:,:,4])

        confidence_loss = tf.square(tf.subtract(top_iou[:,:,:,:,0], matching_boxes[:,:,:,:,4]))

        self.loss_layers['confidence_loss'] = confidence_loss

        print("conf_loss", confidence_loss.shape)

        object_recognition = tf.multiply(tf.cast(obj, tf.float32), confidence_loss)

        self.loss_layers['object_recognition'] = object_recognition

        self.loss_obj = cfg.obj_weight * tf.reduce_sum(object_recognition)

        noobject_recognition = tf.multiply(tf.cast(noobj, tf.float32), confidence_loss)

        self.loss_layers['noobject_recognition'] = noobject_recognition

        self.loss_noobj = cfg.noobj_weight * tf.reduce_sum(
            noobject_recognition
        )

        #true_classes = tf.argmax(self.train_object_recognition, -1)

        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_classes,
                                                labels=self.train_object_recognition)

        print("class_loss", class_loss.shape)

        obj_classes = tf.tile(obj,
                            [1, 1, 1, 10])

        class_loss = tf.multiply(tf.cast(obj_classes, tf.float32), class_loss)

        self.loss_class = (tf.reduce_sum(class_loss) * cfg.class_weight)

        self.loss = self.loss_position + \
                    self.loss_dimension + \
                    self.loss_obj + \
                    self.loss_noobj + \
                    self.loss_class

        self.bool = self.loss_obj

    def get_network(self):
        return self.network

