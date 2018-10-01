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
from data_loader import load_files, disable_transformation

disable_transformation()

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
        images, _, _ = load_files(sys.argv[1:])

        img = images[0]

        #normalise data  between 0 and 1
        imgs = np.array(images)/127.5-1

        boxes = sess.run(yolo.output, feed_dict={
            yolo.x: imgs,
            yolo.anchors: anchors,
        })

        proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=True).tolist()[0]




        proc_boxes.sort(key=lambda box: -box[5])

        #proc_boxes


        trim_overlap = True

        i=0
        while i < len(proc_boxes):
            box = proc_boxes[i]
            if box[5] < cfg.object_detection_threshold:
                del proc_boxes[i]
            else:
                x, y, w, h = (box[1],box[2],box[3],box[4])
                box[1] = x - w/2
                box[2] = y - h/2
                box[3] = x + w/2
                box[4] = y + h/2
                i = i + 1
        i = 0


        while i < len(proc_boxes)-1 and trim_overlap:
            box = proc_boxes[i]
            j = i+1
            while j < len(proc_boxes):
                box2 = proc_boxes[j]

                xA = max(box[1], box2[1])
                yA = max(box[2], box2[2])
                xB = min(box[3], box2[3])
                yB = min(box[4], box2[4])

                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)


                boxAArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

                iou = interArea / float(boxAArea + boxBArea - interArea)

                if iou > 0.8:
                    if (box[5] >= box2[5]):
                        del proc_boxes[j]
                        j = j-1
                    else:
                        del proc_boxes[i]
                        i = i-1
                        break
                j = j + 1
            i = i+1

        for box in proc_boxes:
            height, width = img.shape[:2]

            cls = yolo.names[int(box[0])]

            hex = cls.encode('utf-8').hex()[0:6]

            color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

            color = [255, 255, 255]

            if (box[5]>cfg.object_detection_threshold):
                print(box)

                x1 = max(int(width*box[1]), 0)
                y1 = max(int(height*box[2]), 0)
                x2 = int(width*box[3])
                y2 = int(height*box[4])

                cv2.rectangle(img, (x1, y1),
                              (x2, y2),
                              (color[0], color[1], color[2]), 3+int(7*box[4]), 8)

        for box in proc_boxes:
            cls = yolo.names[int(box[0])]

            hex = cls.encode('utf-8').hex()[0:6]

            color = tuple(int(hex[k:k+2], 16) for k in (0, 2 ,4))

            if (box[5]>cfg.object_detection_threshold):
                height, width = img.shape[:2]

                color = [255, 255, 255]

                avg_col = color[0] + color[1] + color[2]

                text_col = (255, 255, 255)

                if avg_col > 127:
                    text_col = (0, 0, 0)

                x1 = max(int(width*box[1]), 0)
                y1 = max(int(height*box[2]), 0)
                x2 = int(width*box[3])
                y2 = int(height*box[4])

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
