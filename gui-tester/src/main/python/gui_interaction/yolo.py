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

    def prediction_layer(self, inputs, anchors):
        anchors_size = anchors.shape.as_list()[0]
        classes = len(self.names)

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(inputs.shape[1]), [inputs.shape[2]]),
                                        (1, inputs.shape[2], inputs.shape[1], 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [tf.shape(self.x)[0], 1, 1, int(anchors_size/3), 1])

        height, width = inputs.shape.as_list()[1:3]

        stride = (cfg.width // height, cfg.height // width)

        preds = tf.reshape(
            self.conv2d(inputs, (classes + 5) * anchors_size, kernals=1, strides=1),
            [-1, inputs.shape[1], inputs.shape[2], anchors_size, classes+5])

        anchors_weight = tf.tile(
            tf.reshape(anchors, [1, 1, 1, anchors_size, 2]),
            [tf.shape(preds)[0], inputs.shape[1], inputs.shape[2], 1, 1]
        )

        box_xy = tf.multiply(tf.nn.sigmoid(preds[...,:2]) + cell_grid, stride)
        box_wh = tf.exp(preds[...,2:4]) * anchors_weight * stride
        box_conf = tf.expand_dims(tf.nn.sigmoid(preds[...,4]), dim=-1)
        box_classes = preds[..., 5:]

        return tf.identity(tf.reshape(tf.concat([box_xy, box_wh, box_conf, box_classes], axis=-1),
                                      [-1, inputs.shape[1] * inputs.shape[2] * anchors_size, classes+5]))

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
        preds1 = self.prediction_layer(residual, self.anchors[:3])

        self.network = self.conv2d(self.network, 256, kernals=1)

        self.network = self.yolo_upsample(self.network, tf.shape(reorg2))
        print(self.network.shape)
        self.network = tf.concat([self.network, reorg2], axis=-1)

        self.network, residual = self.yolo_layers(self.network, 256)

        preds2 = self.prediction_layer(residual, self.anchors[3:6])

        self.network = self.conv2d(self.network, 128, kernals=1)

        self.network = self.yolo_upsample(self.network, tf.shape(reorg))
        print(self.network.shape)
        self.network = tf.concat([self.network, reorg], axis=-1)

        _, residual = self.yolo_layers(self.network, 128)
        preds3 = self.prediction_layer(residual, self.anchors[6:9])


        self.network = tf.concat([preds1, preds2, preds3], axis=1)

        self.output = self.network

        print("Network shape:", self.network.shape)


    def create_training(self):

        classes = len(self.names)
        anchors = int(len(cfg.anchors)/2)

        self.train_bounding_boxes = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 5 + classes], "train_bb")

        self.iou_threshold = tf.placeholder(tf.float32)

        truth = tf.reshape(self.train_bounding_boxes, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5 + classes])

        pred_boxes = self.output

        pred_confidence = tf.reshape(
            pred_boxes[..., 4],
            [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 1]
        )

        pred_classes = pred_boxes[..., 5:]



        pred_boxes_xy = (pred_boxes[..., :2])

        epsilon = tf.constant(self.epsilon)

        pred_boxes_wh = pred_boxes[..., 2:4]

        truth_boxes_xy = truth[..., :2] * (cfg.grid_shape[0]/width)
        truth_boxes_wh = truth[..., 2:4] * (cfg.grid_shape[1]/height)

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

        union_areas = pred_areas + true_areas - intersect_areas + tf.ones_like(intersect_areas)

        self.loss_layers['intersect_areas'] = intersect_areas
        self.loss_layers['union_areas'] = union_areas
        self.loss_layers['true_areas'] = true_areas
        self.loss_layers['pred_areas'] = pred_areas

        iou = tf.truediv(intersect_areas, union_areas)
        self.loss_layers['raw_iou'] = iou

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



        new_indices = tf.reshape(new_indices, [-1, 3])

        self.loss_layers['new_indices'] = new_indices

        iou_reshaped = tf.reshape(iou, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5, 1])

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

        self.best_iou = top_iou

        self.loss_layers['top_iou'] = top_iou
        self.loss_layers['best_iou'] = matching_boxes

        obj = tf.cast(truth[..., 4], tf.float32)

        self.loss_layers['obj'] = obj

        print("obj:", obj.shape)

        obj_xy = tf.reshape(tf.tile(tf.expand_dims(obj * cfg.coord_weight, axis=3),[ 1, 1, 1, anchors, 2]),
                            [-1, cfg.grid_shape[0] * cfg.grid_shape[1], anchors, 2])

        print("obj_xy:", obj_xy.shape)

        self.loss_layers['obj_xy'] = obj_xy

        pos_mask_count = tf.reduce_sum(tf.cast(obj_xy>0, tf.float32)) + epsilon

        total_pos_loss = tf.reshape(tf.square(pred_boxes[...,0:2] - truth_tiled[...,0:2]), [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5, 2]) \
                         / pos_mask_count
        total_dim_loss = tf.reshape(tf.square(tf.sqrt(pred_boxes[...,2:4]) - tf.sqrt(truth_tiled[...,2:4])), [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5, 2]) \
                         / pos_mask_count

        print("total pos loss:", total_pos_loss.shape)

        self.loss_position = tf.reduce_sum(obj_xy * total_pos_loss)

        self.loss_dimension = tf.reduce_sum(obj_xy * total_dim_loss)

        obj_conf = tf.cast(tf.reshape(top_iou,
                                 [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1]) < cfg.object_detection_threshold, tf.float32) * \
              (1 - truth[...,4]) * cfg.noobj_weight

        obj_conf = obj_conf + truth[...,4] * cfg.obj_weight

        obj_conf = tf.reshape(tf.tile(tf.expand_dims(obj_conf, axis=3),[ 1, 1, 1, anchors, 1]),
                            [-1, cfg.grid_shape[0] * cfg.grid_shape[1], anchors, 1])

        total_conf_loss = tf.reshape(tf.square(pred_boxes[...,4] - truth_tiled[...,4]), [-1, cfg.grid_shape[0] * cfg.grid_shape[1] , 5, 1]) \
                          / (tf.reduce_sum(tf.cast(obj_conf>0, tf.float32)) + epsilon)

        self.loss_layers['confidence_loss'] = total_conf_loss

        print("conf_loss", total_conf_loss.shape)

        object_recognition = tf.multiply(tf.cast(obj_conf, tf.float32), total_conf_loss)

        self.loss_layers['object_recognition'] = object_recognition

        self.loss_obj = tf.reduce_sum(object_recognition)

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_classes,
                                                labels=tf.cast(self.train_object_recognition, tf.int32))

        print("class_loss", class_loss.shape)

        obj_classes = tf.reshape(obj, [-1, cfg.grid_shape[0], cfg.grid_shape[1]])#tf.tile(obj, [1, 1, 1, 10])

        class_loss = tf.multiply(obj_classes, class_loss) * cfg.class_weight \
                     / (tf.reduce_sum(tf.cast(obj_classes>0, tf.float32))+epsilon)

        self.loss_class = tf.reduce_sum(class_loss)

        self.loss = self.loss_position + self.loss_dimension + self.loss_obj + self.loss_class


        obj_sens = tf.reshape(tf.cast(truth[...,4]>0, tf.float32), [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

        class_assignments = tf.argmax(self.pred_classes,-1)

        print(class_assignments.shape)
        print(self.train_object_recognition.shape)

        correct_classes = tf.cast(tf.equal(class_assignments, tf.cast(self.train_object_recognition, tf.int64)), tf.float32)

        identified_objects_tpos = tf.reshape(tf.cast(top_iou >= self.iou_threshold, tf.float32),
                                            [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * tf.reshape(
            tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
            [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

        identified_objects_fpos = tf.maximum((1-obj_sens) + (tf.reshape(tf.cast(top_iou < self.iou_threshold, tf.float32),
                                             [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * obj_sens), 1) * \
                                             tf.reshape(
            tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
            [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

        identified_objects_fneg =  tf.reshape(
            tf.cast(matching_boxes[..., 4] < self.object_detection_threshold, tf.float32),
            [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

        self.average_iou = tf.reduce_sum(top_iou) / (tf.reduce_sum(tf.cast(truth[...,4]>0, tf.float32)) + epsilon)


        self.matches = obj_sens * identified_objects_tpos

        self.true_positives = tf.reduce_sum(obj_sens * identified_objects_tpos
                                            )
        self.false_positives = tf.reduce_sum(identified_objects_fpos)
        self.false_negatives = tf.reduce_sum(obj_sens * identified_objects_fneg)

        self.true_negatives = tf.reduce_sum((1 - obj_sens) * identified_objects_fneg)

        self.mAP = 0

        for i in range(10):
            identified_obj_tpos = tf.reshape(tf.cast(top_iou >= (0.5+(i * 0.05)), tf.float32),
                                [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * tf.reshape(
                tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
                [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

            identified_obj_fpos = tf.reshape(
                tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
                [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

            true_p = tf.reduce_sum(obj_sens * identified_obj_tpos
                                                )
            false_p = tf.reduce_sum((1-obj_sens) * identified_obj_fpos)
            self.mAP = self.mAP + (((true_p+1)/(true_p+false_p+1))/10)

        if (cfg.enable_logging):
            tf.summary.histogram("loss_position", total_pos_loss)
            tf.summary.histogram("loss_dimension", total_dim_loss)
            tf.summary.histogram("loss_object", object_recognition)
            tf.summary.histogram("loss_classification", class_loss)
            tf.summary.scalar("loss_pos", self.loss_position)
            tf.summary.scalar("loss_dim", self.loss_dimension)
            tf.summary.scalar("loss_obj", self.loss_obj)
            tf.summary.scalar("loss_class", self.loss_class)
            tf.summary.scalar("total_loss", self.loss)


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
