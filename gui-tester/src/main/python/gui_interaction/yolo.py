import numpy as np
import tensorflow as tf
import config as cfg
import numpy as np

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
    loss_class = None
    epsilon = 1E-8
    output = None
    pred_boxes = None
    pred_classes = None
    loss_layers = {}
    cell_grid = None
    layer_counter = 0
    false_positives = None
    false_negatives = None
    true_positives = None
    true_negatives = None
    matches = None
    iou_threshold = None
    mAP = None
    object_detection_threshold = cfg.object_detection_threshold
    average_iou = None
    network_predictions = []

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

    def create_filter(self, size, name):

        var = tf.Variable(tf.random_normal(size, stddev=cfg.var_sd), name=name)
        self.variables.append(var)
        return var
        #
        # return tf.get_variable(name, size, initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
        #                    dtype=tf.float32)

    def leaky_relu(self, layer):
        self.layer_counter = self.layer_counter + 1
        x = tf.layers.batch_normalization(layer, training=self.is_training, name="batch_norm" + str(self.layer_counter))
        return tf.maximum(x, 0.1 * x)

    def conv2d(self, input, filters, kernals=3, strides=1):
        return self.leaky_relu(tf.layers.conv2d(input, filters, kernals, strides, padding="same", name="conv2d" + str(self.layer_counter)))

    def darknet53_layers(self, input, filters):
        layer = self.conv2d(input, filters, kernals=1)
        layer = self.conv2d(layer, filters * 2)
        return layer + input

    def reshape(self, arr, scale, width, height, anchors):

        indices = []

        print(arr.shape)
        for i in range(width):
            ind = []
            for j in range(height):
                inh = []
                for s in range(scale):
                    for t in range(scale):
                        for a in range(anchors):
                            inh.append([(i*scale)+t, (j*scale)+s, a])
                ind.append(inh)

            indices.append(ind)

        indices = np.array(indices, np.int32)

        print(indices.shape)

        #indices = tf.expand_dims(indices, 0)

        #indices = tf.tile(indices, [tf.shape(arr)[0], 1, 1, 1])

        return tf.map_fn(lambda x: tf.gather_nd(x, indices), arr)
        #return arr


    def prediction_layer(self, inputs, anchors, scale):
        anchors_size = anchors.shape.as_list()[0]
        classes = len(self.names)

        height, width = inputs.shape.as_list()[1:3]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(width), [height]),
                                        (1, height, width, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [tf.shape(inputs)[0], 1, 1, anchors_size, 1])

        preds = tf.reshape(
            self.conv2d(inputs, (classes + 5) * anchors_size, kernals=1, strides=1),
            [-1, inputs.shape[1], inputs.shape[2], anchors_size, classes+5])

        anchors_weight = tf.tile(
            tf.reshape(anchors, [1, 1, 1, anchors_size, 2]),
            [tf.shape(preds)[0], inputs.shape[1], inputs.shape[2], 1, 1]
        )

        box_xy = (tf.nn.sigmoid(preds[...,:2]) + cell_grid)/scale
        box_wh = tf.exp(preds[...,2:4]) * anchors_weight
        box_conf = tf.expand_dims(tf.nn.sigmoid(preds[...,4]), axis=-1)
        box_classes = preds[..., 5:]

        upscaled_box_num = anchors_size * scale * scale

        layer = tf.concat([box_xy, box_wh, box_conf, box_classes], axis=-1)

        print(layer.shape)

        if (scale > 1):
            layer = self.reshape(layer, scale, int(width/scale), int(height/scale), anchors_size)

            print(layer.shape)

        return tf.identity(tf.reshape(
            layer,
            [-1, int(height/scale), int(width/scale), upscaled_box_num, classes+5]))

    def yolo_layers(self, input, filters):
        layer = self.conv2d(input, filters, kernals=1)
        layer = self.conv2d(layer, filters*2, kernals=3)
        layer = self.conv2d(layer, filters, kernals=1)
        layer = self.conv2d(layer, filters*2, kernals=3)
        layer = self.conv2d(layer, filters, kernals=1)
        residual = layer
        layer = self.conv2d(layer, filters*2, kernals=3)
        return residual, layer

    def yolo_upsample(self, input, output):
        layer = tf.pad(input, [[0, 0], [1, 1],
                            [1, 1], [0, 0]], mode="SYMMETRIC")

        layer = tf.image.resize_nearest_neighbor(layer, (output[2], output[1]))

        return tf.identity(layer)

    def create_network(self):

        anchors_size = int(len(cfg.anchors)/2) # n / 2 points * 3 sample sizes
        classes = len(self.names)

        self.x = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1], "input")

        self.object_detection_threshold = tf.placeholder(tf.float32)

        self.anchors = tf.placeholder(tf.float32, [anchors_size, 2], "anchors")

        #construct darknet-53 layers
        print("Starting Darknet-53 layers")
        self.network = self.conv2d(self.x, 32)
        self.network = self.conv2d(self.network, 64, strides=2)

        self.network = self.darknet53_layers(self.network, 32)
        self.network = self.conv2d(self.network, 128, strides=2)

        for i in range(2):
            self.network = self.darknet53_layers(self.network, 64)

        self.network = self.conv2d(self.network, 256, strides=2)

        for i in range(8):
            self.network = self.darknet53_layers(self.network, 128)

        reorg = self.network

        self.network = self.conv2d(self.network, 512, strides=2)

        for i in range(8):
            self.network = self.darknet53_layers(self.network, 256)

        reorg2 = self.network

        self.network = self.conv2d(self.network, 1024, strides=2)

        for i in range(4):
            self.network = self.darknet53_layers(self.network, 512)


        #begin YOLOv3 layers
        print("Starting YOLOv3 layers.")

        self.network, residual = self.yolo_layers(self.network, 512)
        preds1 = self.prediction_layer(residual, self.anchors[:3], 1)

        self.network = self.conv2d(self.network, 256, kernals=1)

        self.network = self.yolo_upsample(self.network, tf.shape(reorg2))
        print(self.network.shape)
        self.network = tf.concat([self.network, reorg2], axis=-1)

        self.network, residual = self.yolo_layers(self.network, 256)

        preds2 = self.prediction_layer(residual, self.anchors[3:6], 2)

        self.network = self.conv2d(self.network, 128, kernals=1)

        self.network = self.yolo_upsample(self.network, tf.shape(reorg))
        print(self.network.shape)
        self.network = tf.concat([self.network, reorg], axis=-1)

        _, residual = self.yolo_layers(self.network, 128)
        preds3 = self.prediction_layer(residual, self.anchors[6:9], 4)

        print("Preds shape:", preds1.shape, preds2.shape, preds3.shape)

        self.network_predictions = [preds1, preds2, preds3]

        self.network = tf.concat([preds1, preds2, preds3], axis=3)

        self.output = self.network

        print("Network shape:", self.network.shape)


    def create_training(self):

        classes = len(self.names)
        anchors = int(len(cfg.anchors)/2)

        self.train_bounding_boxes = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 5 + classes], "train_bb")
        truth = tf.reshape(self.train_bounding_boxes,
                           [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1, 5 + classes])
        self.iou_threshold = tf.placeholder(tf.float32)

        pred_boxes = self.output

        pred_classes = pred_boxes[..., 4:]



        pred_boxes_xy = pred_boxes[..., :2]

        epsilon = tf.constant(self.epsilon)

        pred_boxes_wh = pred_boxes[..., 2:4]

        truth_tiled = tf.tile(truth, [1, 1, 1, pred_boxes.shape[3], 1])

        truth_boxes_xy = truth_tiled[..., :2]
        truth_boxes_wh = truth_tiled[..., 2:4]

        pred_wh_half = pred_boxes_wh/2
        pred_min = pred_boxes_xy - pred_wh_half
        pred_max = pred_boxes_xy + pred_wh_half

        true_wh_half = truth_boxes_wh / 2
        true_min = truth_boxes_xy - true_wh_half
        true_max = truth_boxes_xy + true_wh_half

        intersect_mins = tf.maximum(pred_min,  true_min)
        intersect_maxes = tf.minimum(pred_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = truth_boxes_wh[..., 0] * truth_boxes_wh[..., 1]
        pred_areas = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas

        self.loss_layers['intersect_areas'] = intersect_areas
        self.loss_layers['union_areas'] = union_areas
        self.loss_layers['true_areas'] = true_areas
        self.loss_layers['pred_areas'] = pred_areas

        iou = tf.truediv(intersect_areas, union_areas + epsilon)
        self.loss_layers['raw_iou'] = iou

        top_iou = tf.reshape(tf.reduce_max(iou, axis=-1),
                    [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1])

        top_iou = tf.tile(top_iou,
                          [1, 1, 1, iou.shape[3]])

        iou_mask = tf.cast(tf.equal(iou, top_iou), tf.float32)

        ignore_mask = tf.cast(iou > 0.5, tf.float32)

        ignore_mask = 1 - (ignore_mask * (1-iou_mask))

        iou_mask = tf.expand_dims(iou_mask, -1)

        box_preds = self.output.shape[3]

        print("IOU:", iou.shape, top_iou.shape, iou_mask.shape)

        obj = tf.cast(truth_tiled[..., 4], tf.float32)

        print(obj.shape)

        self.loss_layers['obj'] = obj

        print("obj:", obj.shape)

        obj_xy = tf.tile(tf.expand_dims(obj * cfg.coord_weight, axis=-1),
                                    [ 1, 1, 1, 1, 2]) * \
                 tf.tile(iou_mask, [1, 1, 1, 1, 2])

        print("obj_xy:", obj_xy.shape)

        self.loss_layers['obj_xy'] = obj_xy

        #pos_mask_count = tf.reduce_sum(tf.cast(obj_xy>0, tf.float32)) + epsilon

        total_pos_loss = tf.square(obj_xy*(truth_tiled[...,0:2] - pred_boxes[...,0:2]))

        total_dim_loss = tf.square(obj_xy*(tf.sqrt(epsilon + truth_tiled[...,2:4]) - tf.sqrt(epsilon + pred_boxes[...,2:4])))


        print("total pos loss:", total_pos_loss.shape)

        self.loss_position = tf.reduce_sum(total_pos_loss)

        #Issue here: When assinging a pred box to match a true box, if the box is in a smaller grid cell it
        #can never reach its intended x,y positions!
        self.loss_dimension = tf.reduce_sum(total_dim_loss)

        conf_diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_boxes[...,4],
                                                            labels=truth_tiled[...,4])

        conf_loss = (1 - obj) * cfg.noobj_weight * conf_diff

        conf_loss = conf_loss + obj * conf_diff * ignore_mask * cfg.obj_weight

        total_conf_loss = conf_loss

        self.loss_layers['confidence_loss'] = total_conf_loss

        print("conf_loss", total_conf_loss.shape)

        self.loss_obj = tf.reduce_sum(total_conf_loss)

        self.train_object_recognition = tf.tile(truth[..., 4:],
                [1, 1, 1, pred_classes.shape[3], 1])

        class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_classes,
                                                labels=self.train_object_recognition)

        print("class_loss", class_loss.shape)

        obj_classes =tf.tile(tf.expand_dims(obj, -1) * iou_mask, [1, 1, 1, 1, classes+1])

        class_loss = tf.multiply(obj_classes, class_loss) * cfg.class_weight

        self.loss_class = tf.reduce_sum(class_loss)

        self.loss = self.loss_position + self.loss_dimension + self.loss_class # + self.loss_obj


    def get_network(self):
        return self.network

    def convert_net_to_bb(self, boxes, filter_top=True):
        b_boxes = []

        for image in range(boxes.shape[0]):
            for i in range(cfg.grid_shape[0]):
                for j in range(cfg.grid_shape[1]):
                    cell = boxes[image][j][i]
                    classes = cell[int((len(cfg.anchors)/2)*5):]
                    amax = np.array([np.argmax(classes)])

                    plot_box = [0, 0, 0, 0, 0]

                    for k in range(int(len(cfg.anchors)/2)):
                        box = cell[k*5:(k+1)*5]

                        if not filter_top:
                            box[0] = box[0]/cfg.grid_shape[0]
                            box[1] = box[1]/cfg.grid_shape[1]
                            box[2] = box[2]/cfg.grid_shape[0]
                            box[3] = box[3]/cfg.grid_shape[1]
                            box_p = np.append(amax, box)
                            b_boxes.append(box_p)
                        elif (box[4]>=plot_box[4]):
                            plot_box = box
                            plot_box[0] = plot_box[0]/cfg.grid_shape[0]
                            plot_box[1] = plot_box[1]/cfg.grid_shape[1]
                            plot_box[2] = plot_box[2]/cfg.grid_shape[0]
                            plot_box[3] = plot_box[3]/cfg.grid_shape[1]


                    if filter_top:
                        plot_box = np.append(amax, plot_box)
                        b_boxes.append(plot_box)

        return np.array(b_boxes)
