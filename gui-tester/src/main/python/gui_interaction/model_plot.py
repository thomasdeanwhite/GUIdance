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

        height, width = image.shape[:2]

        aspect = height/width

        print(image.shape)

        if aspect > 1: # portrait
            padding = round((height-width)/2)
            for i in range(padding):
                elements = np.transpose(np.expand_dims(np.zeros([image.shape[0]]), 0))
                image = np.append(elements, image, 1)
            for i in range(padding):
                elements = np.transpose(np.expand_dims(np.zeros([image.shape[0]]), 0))
                image = np.append(image, elements, 1)
        else: #landscape
            padding = round((width-height)/2)
            for i in range(padding):
                elements = np.transpose(np.expand_dims(np.zeros([image.shape[1]]), 1))
                image = np.append(elements, image, 0)
            for i in range(padding):
                elements = np.transpose(np.expand_dims(np.zeros([image.shape[1]]), 1))
                image = np.append(image, elements, 0)

        image = np.uint8(cv2.resize(image, (cfg.width, cfg.height)))
        image = np.reshape(image, [cfg.width, cfg.height, 1])



        images.append([image, img_raw])

    return images



if __name__ == '__main__':

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

    gpu_options = tf.GPUOptions(allow_growth=True)

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

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

        boxes = sess.run(yolo.boxes, feed_dict={
            yolo.x: imgs,
            yolo.anchors: anchors,
        })

        #print(boxes[..., 2:4])

        boxes[..., 2:4] = np.sqrt(boxes[..., 2:4])

        proc_boxes = boxes#[..., :5]

        # for x in range(13):
        #     for y in range(13):
        #         b = proc_boxes[0, x, y]
        #         print("(", x, ",", y, ")", "x:", np.max(b[3:15,0]), "-", np.min(b[3:15,0]),
        #         "y:", np.max(b[...,1]), "-", np.min(b[...,1]))

        proc_boxes[...,0:4] = proc_boxes[...,0:4]/cfg.grid_shape[0]

        #proc_boxes[..., 2:4] = 0.2

        proc_boxes = np.reshape(proc_boxes, [proc_boxes.shape[0], -1, 5+len(yolo.names)])

        #proc_boxes[..., :4] = np.max(0.0, np.min(1.0, proc_boxes[..., :4]))


        print(proc_boxes)

        #proc_boxes = np.extract(proc_boxes[...,4] > cfg.object_detection_threshold, proc_boxes)

        proc_boxes = proc_boxes[0].tolist()

        # classes = np.reshape(np.argmax(proc_boxes[...,5:], axis=-1), [-1, proc_boxes.shape[1], 1])
        #
        #
        # classes = np.reshape(np.argmax(proc_boxes[...,5:], axis=-1), [-1, proc_boxes.shape[1], 1])
        #
        # proc_boxes = proc_boxes[..., 0:5]
        #
        # proc_boxes = np.append(classes, proc_boxes, axis=-1).tolist()[0]

        img = images[0][1]

        proc_boxes.sort(key=lambda box: -box[4])

        i=0

        while i < len(proc_boxes):
            box = proc_boxes[i]
            if box[4] < cfg.object_detection_threshold:
                del proc_boxes[i]
            else:
                x, y, w, h = (box[0],box[1],box[2],box[3])
                box[0] = x - w/2
                box[1] = y - h/2
                box[2] = x + w/2
                box[3] = y + h/2
                i = i + 1

        i=0

        while i < len(proc_boxes)-1:
            box = proc_boxes[i]
            j = i+1
            while j < len(proc_boxes):
                box2 = proc_boxes[j]

                xA = max(box[0], box2[0])
                yA = max(box[1], box2[1])
                xB = min(box[2], box2[2])
                yB = min(box[3], box2[3])

                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)


                boxAArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

                iou = interArea / float(boxAArea + boxBArea - interArea)

                if iou > 0.8:
                    if (box[4] >= box2[4]):
                        del proc_boxes[j]
                        j = j-1
                    else:
                        del proc_boxes[i]
                        i = i-1
                        break
                j = j + 1
            i = i+1


        for box in proc_boxes:
            height, width, channels = img.shape

            classes = ""

            display = False

            for i in range(len(yolo.names)):
                if (box[5+i]*box[4] >= cfg.object_detection_threshold):
                    classes += yolo.names[i] + ","
                    display = True

            if (display):
                cls = classes

                hex = cls.encode('utf-8').hex()[0:6]

                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))


                print(box)

                x1 = max(int(width*(box[0])), 0)
                y1 = max(int(height*(box[1])), 0)
                x2 = int(width*((box[2])))
                y2 = int(height*(box[3]))

                cv2.rectangle(img, (x1, y1),
                              (x2, y2),
                              (color[0], color[1], color[2]), 3+int(7*box[4]), 8)

        for box in proc_boxes:

            classes = ""
            desc = ""

            display = False

            max_conf = 0

            for i in range(len(yolo.names)):
                conf = box[4]*box[5+i]
                if (conf >= cfg.object_detection_threshold):
                    if (conf > max_conf):
                        classes = yolo.names[i]
                        max_conf = conf

                    desc += yolo.names[i] + " "
                    display = True

            desc += str(int(max_conf*100))

            if (display):
                cls = classes

                hex = cls.encode('utf-8').hex()[0:6]

                color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

                height, width, channels = img.shape

                avg_col = color[0] + color[1] + color[2]

                text_col = (255, 255, 255)

                if avg_col > 127:
                    text_col = (0, 0, 0)

                    x1 = max(int(width*(box[0])), 0)
                    y1 = max(int(height*(box[1])), 0)
                    x2 = int(width*((box[2])))
                    y2 = int(height*(box[3]))

                cv2.rectangle(img,
                              (x1, y1-int(10*box[4])-15),
                              (x1 + (5 + len(cls))*7, y1),
                              (color[0], color[1], color[2]), -1, 8)

                cv2.putText(img, desc,
                            (x1, y1-int(10*box[4])-2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, text_col, 1)

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
