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

    def create_network(self):

        anchors_size = int(len(cfg.anchors)/2)
        classes = len(self.names)

        self.x = tf.placeholder(tf.float32, [None, cfg.height, cfg.width, 1], "input")

        self.object_detection_threshold = tf.placeholder(tf.float32)

        self.anchors = tf.placeholder(tf.float32, [anchors_size, 2], "anchors")

        self.network = self.leaky_relu(tf.layers.conv2d(self.x, 32, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = tf.layers.max_pooling2d(self.network, 2, 2, name="max2d" + str(self.layer_counter))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n1", self.network)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 64, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n1.5", self.network)

        self.network = tf.layers.max_pooling2d(self.network, 2, 2, name="max" + str(self.layer_counter))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n2", self.network)


        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 128, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 64, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 128, 3, padding="same", name="conv2d" + str(self.layer_counter)))

        self.network = tf.layers.max_pooling2d(self.network, 2, 2, name="max" + str(self.layer_counter))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n3", self.network)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 256, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)


        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 128, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 256, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        self.network = tf.layers.max_pooling2d(self.network, 2, 2, name="max" + str(self.layer_counter))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n4", self.network)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 512, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 256, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 512, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 256, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 512, 3, padding="same", name="conv2d" + str(self.layer_counter)))

        reorg = tf.extract_image_patches(self.network, [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME", name="extract" + str(self.layer_counter))

        print("reorg:", reorg.shape)

        self.network = tf.layers.max_pooling2d(self.network, 2, 2, name="max" + str(self.layer_counter))
        print(self.network.shape)

        if cfg.enable_logging:
            tf.summary.histogram("n5", self.network)
            #tf.summary.histogram("reorg", reorg)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 512, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 512, 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        print("Combining ", self.network.shape, "with reorg:", reorg.shape)
        self.network = tf.concat([self.network, reorg], axis=-1, name="concat" + str(self.layer_counter))

        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, 1024, 3, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        self.network = self.leaky_relu(tf.layers.conv2d(self.network, int(anchors_size*5 + classes), 1, padding="same", name="conv2d" + str(self.layer_counter)))
        print(self.network.shape)

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(cfg.grid_shape[0]), [cfg.grid_shape[1]]),
                                        (1, cfg.grid_shape[1], cfg.grid_shape[0], 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        predictions = tf.reshape(self.network, [-1, cfg.grid_shape[0] * cfg.grid_shape[1], int(anchors_size*5 + classes)])

        raw_boxes = predictions[:, :, 0:(anchors_size*5)]

        pred_boxes = tf.reshape(raw_boxes, [-1,  cfg.grid_shape[0], cfg.grid_shape[1], anchors_size, 5])

        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [tf.shape(pred_boxes)[0], 1, 1, anchors_size, 1])

        pred_boxes_xy = tf.sigmoid(pred_boxes[..., 0:2]) + self.cell_grid
        pred_boxes_wh = tf.exp(pred_boxes[..., 2:4])

        anchors_weight = tf.tile(
            tf.reshape(self.anchors, [1, 1, 1, anchors_size, 2]),
            [tf.shape(pred_boxes)[0], cfg.grid_shape[0], cfg.grid_shape[1],
             1, 1])

        pred_boxes_wh = pred_boxes_wh * anchors_weight

        confidence = tf.sigmoid(tf.reshape(pred_boxes[:, :, :, :, 4],
                                 [-1, cfg.grid_shape[0], cfg.grid_shape[1], anchors_size, 1]))

        pred_boxes = tf.concat([pred_boxes_xy, pred_boxes_wh], axis=-1)
        pred_boxes = tf.concat([pred_boxes, confidence], axis=-1)

        self.pred_boxes = pred_boxes

        pred_classes = tf.nn.softmax(tf.reshape(
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

        self.train_object_recognition = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1]], "train_obj_rec")
        self.train_bounding_boxes = tf.placeholder(tf.float32, [None, cfg.grid_shape[0], cfg.grid_shape[1], 5], "train_bb")

        self.iou_threshold = tf.placeholder(tf.float32)

        truth = tf.reshape(self.train_bounding_boxes, [-1, cfg.grid_shape[0], cfg.grid_shape[1], 1, 5])

        pred_boxes = self.pred_boxes
        print("pred:", pred_boxes.shape)

        pred_confidence = tf.reshape(
            pred_boxes[:, :, 4],
            [-1, cfg.grid_shape[0] * cfg.grid_shape[1] * anchors, 1]
        )

        print("conf:", pred_confidence.shape)

        pred_classes = self.pred_classes

        print("p_classes:", pred_classes.shape)



        pred_boxes_xy = (pred_boxes[..., 0:2])

        epsilon = tf.constant(self.epsilon)

        pred_boxes_wh = pred_boxes[..., 2:4]

        truth_tiled = tf.tile(truth, [1, 1, 1, anchors, 1])

        print("truth tiled:", truth_tiled.shape)

        truth_boxes_xy = truth_tiled[..., 0:2]
        truth_boxes_wh = truth_tiled[..., 2:4]


        self.loss_layers['pred_boxes_xy'] = pred_boxes_xy
        self.loss_layers['pred_boxes_wh'] = pred_boxes_wh


        self.loss_layers['truth_boxes_xy'] = truth_boxes_xy
        self.loss_layers['truth_boxes_wh'] = truth_boxes_wh

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
        self.loss_layers['pred_xy'] = pred_boxes_xy
        self.loss_layers['truth_xy'] = truth_boxes_xy

        pos_mask_count = tf.reduce_sum(tf.cast(obj_xy>0, tf.float32)) + epsilon

        total_pos_loss = obj_xy * tf.reshape(tf.losses.mean_squared_error(truth_boxes_xy, pred_boxes_xy, reduction=tf.losses.Reduction.NONE),
                                    [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5, 2]) \
                         / pos_mask_count


        self.loss_layers['pos_loss'] = total_pos_loss

        total_dim_loss = obj_xy * tf.reshape(tf.losses.mean_squared_error(tf.sqrt(truth_boxes_wh + epsilon),
                                                tf.sqrt(pred_boxes_wh + epsilon), reduction=tf.losses.Reduction.NONE),
                                             [-1, cfg.grid_shape[0] * cfg.grid_shape[1], 5, 2]) \
                         / pos_mask_count

        #total_dim_loss = tf.losses.mean_squared_error(truth_boxes_wh, pred_boxes_wh, reduction=tf.losses.Reduction.NONE)

        self.loss_layers['dim_loss'] = total_dim_loss

        print("total pos loss:", total_pos_loss.shape)

        self.loss_position = tf.reduce_sum(total_pos_loss)

        self.loss_dimension = tf.reduce_sum(total_dim_loss)

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

        #object_recognition = tf.losses.sigmoid_cross_entropy(truth_tiled[..., 4], pred_boxes[..., 4])#tf.multiply(tf.cast(obj_conf, tf.float32), total_conf_loss)

        self.loss_layers['object_recognition'] = total_conf_loss

        self.loss_obj = tf.reduce_sum(total_conf_loss)

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_classes,
                                                labels=tf.cast(self.train_object_recognition, tf.int32))

        print("class_loss", class_loss.shape)

        obj_classes = tf.reshape(obj, [-1, cfg.grid_shape[0], cfg.grid_shape[1]])#tf.tile(obj, [1, 1, 1, 10])

        class_loss = tf.multiply(obj_classes, class_loss) * cfg.class_weight \
                     / (tf.reduce_sum(tf.cast(obj_classes>0, tf.float32))+epsilon)

        self.loss_class = tf.reduce_sum(class_loss)

        self.loss = self.loss_position + self.loss_dimension + self.loss_obj + self.loss_class


        obj_sens = tf.reshape(tf.cast(truth[...,4]>0, tf.float32), [-1, cfg.grid_shape[0], cfg.grid_shape[1]])
        #
        class_assignments = tf.argmax(self.pred_classes,-1)
        #
        # print(class_assignments.shape)
        # print(self.train_object_recognition.shape)
        #
        correct_classes = tf.cast(tf.equal(class_assignments, tf.cast(self.train_object_recognition, tf.int64)), tf.float32)
        #
        identified_objects_tpos = tf.reshape(tf.cast(top_iou >= self.iou_threshold, tf.float32),
                                            [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * tf.reshape(
            tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
            [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * correct_classes

        # identified_objects_fpos = tf.maximum((1-obj_sens) + (tf.reshape(tf.cast(top_iou < self.iou_threshold, tf.float32),
        #                                      [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * obj_sens), 1) * \
        #                                      tf.reshape(
        #     tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
        #     [-1, cfg.grid_shape[0], cfg.grid_shape[1]])
        #
        # identified_objects_fneg =  tf.reshape(
        #     tf.cast(matching_boxes[..., 4] < self.object_detection_threshold, tf.float32),
        #     [-1, cfg.grid_shape[0], cfg.grid_shape[1]])

        self.average_iou = tf.constant(0) #tf.reduce_sum(top_iou) / (tf.reduce_sum(tf.cast(truth[...,4]>0, tf.float32)) + epsilon)


        self.matches = obj_sens * identified_objects_tpos

        self.true_positives = tf.constant(0)#tf.reduce_sum(obj_sens * identified_objects_tpos)

        self.false_positives = tf.constant(0)#tf.reduce_sum(identified_objects_fpos)
        self.false_negatives = tf.constant(0)#tf.reduce_sum(obj_sens * identified_objects_fneg)

        self.true_negatives = tf.constant(0)#tf.reduce_sum((1 - obj_sens) * identified_objects_fneg)

        self.mAP = tf.constant(0)

        # for i in range(10):
        #     identified_obj_tpos = tf.reshape(tf.cast(top_iou >= (0.5+(i * 0.05)), tf.float32),
        #                         [-1, cfg.grid_shape[0], cfg.grid_shape[1]]) * tf.reshape(
        #         tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
        #         [-1, cfg.grid_shape[0], cfg.grid_shape[1]])
        #
        #     identified_obj_fpos = tf.reshape(
        #         tf.cast(matching_boxes[..., 4] >= self.object_detection_threshold, tf.float32),
        #         [-1, cfg.grid_shape[0], cfg.grid_shape[1]])
        #
        #     true_p = tf.reduce_sum(obj_sens * identified_obj_tpos
        #                                         )
        #     false_p = tf.reduce_sum((1-obj_sens) * identified_obj_fpos)
        #     self.mAP = self.mAP + (((true_p+1)/(true_p+false_p+1))/10)

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
            bs = []
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
                            bs.append(box_p)
                        elif (box[4]>=plot_box[4]):
                            plot_box = box
                            plot_box[0] = plot_box[0]/cfg.grid_shape[0]
                            plot_box[1] = plot_box[1]/cfg.grid_shape[1]
                            plot_box[2] = plot_box[2]/cfg.grid_shape[0]
                            plot_box[3] = plot_box[3]/cfg.grid_shape[1]


                    if filter_top:
                        plot_box = np.append(amax, plot_box)
                        bs.append(plot_box)

            b_boxes.append(bs)

        return np.array(b_boxes)
