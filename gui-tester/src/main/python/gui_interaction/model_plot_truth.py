import config as cfg
from yolo import Yolo
import cv2
import numpy as np
import sys
from data_loader import load_file_raw, disable_transformation

disable_transformation()

if __name__ == '__main__':

    yolo = Yolo()

    anchors = np.reshape(np.array(cfg.anchors), [-1, 2])
    images, labels, obj_detect = load_file_raw(sys.argv[1])

    img = images

    #normalise data  between 0 and 1
    imgs = np.array(images)/127.5-1
    labels = np.array(labels)
    obj_detect = np.array(obj_detect)


    trim_overlap = True

    proc_boxes = np.reshape(labels, [169, -1])/cfg.grid_shape[0]
    classes = np.reshape(obj_detect, [169, -1])

    proc_boxes = np.append(classes, proc_boxes, axis=1).tolist()

    i=0
    while i < len(proc_boxes):
        box = proc_boxes[i]
        x, y, w, h = (box[1],box[2],box[3],box[4])
        box[1] = x - w/2
        box[2] = y - h/2
        box[3] = x + w/2
        box[4] = y + h/2
        i = i + 1


    for bc in range(len(proc_boxes)):
        height, width = img.shape[:2]

        box = proc_boxes[bc]

        cls = yolo.names[int(box[0])]

        hex = cls.encode('utf-8').hex()[0:6]

        color = tuple(int(int(hex[k:k+2], 16)*0.75) for k in (0, 2 ,4))

        if (box[5]>0):
            print(box)

            x1 = max(int(width*box[1]), 0)
            y1 = max(int(height*box[2]), 0)
            x2 = int(width*box[3])
            y2 = int(height*box[4])

            cv2.rectangle(img, (x1, y1),
                          (x2, y2),
                          (color[0], color[1], color[2]), 5, 8)

    for bc in range(len(proc_boxes)):
        box = proc_boxes[bc]

        cls = yolo.names[int(box[0])]

        hex = cls.encode('utf-8').hex()[0:6]

        color = tuple(int(int(hex[k:k+2], 16)*0.75) for k in (0, 2 ,4))

        if (box[5]>0):
            height, width = img.shape[:2]

            avg_col = (color[0] + color[1] + color[2]) / 3

            text_col = (255, 255, 255)

            if avg_col > 127:
                text_col = (0, 0, 0)

            x1 = max(int(width*box[1]), 0)
            y1 = max(int(height*box[2]), 0)
            x2 = int(width*box[3])
            y2 = int(height*box[4])

            cv2.rectangle(img,
                          (x1-3, y1-23),
                          (x1 + (len(cls)+1)*10, y1),
                          (color[0], color[1], color[2]), -1, 8)

            cv2.putText(img, cls.upper(),
                        (x1, y1-6),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, text_col, 1, lineType=cv2.LINE_AA)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

