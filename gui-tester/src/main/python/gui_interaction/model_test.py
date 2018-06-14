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
import pyautogui
from operator import itemgetter
import time
import subprocess
import Xlib

def get_window_size(window_name):
    display = Xlib.display.Display()
    root = display.screen().root

    windowIDs = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value
    wid = 0
    win = None
    for windowID in windowIDs:
        window = display.create_resource_object('window', windowID)
        name = window.get_wm_name() # Title
        if isinstance(name, str) and window_name in name:
            wid = windowID
            win = window
            #prop = window.get_full_property(display.intern_atom('_NET_WM_PID'), Xlib.X.AnyPropertyType)
            #pid = prop.value[0] # PID
            break;

    geom = win.get_geometry()

    app_x, app_y, app_w, app_h = (geom.x, geom.y, geom.width, geom.height)

    parent_win = win.query_tree().parent

    while parent_win != 0:
        #print(parent_win)
        p_geom = parent_win.get_geometry()
        app_x += p_geom.x
        app_y += p_geom.y
        parent_win = parent_win.query_tree().parent
    return app_x, app_y, app_w, app_h

if __name__ == '__main__':

    program_name = "Firefox"

    with tf.device(cfg.gpu):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            yolo = Yolo()

            yolo.create_network()

            yolo.set_training(True)
            yolo.set_update_ops(update_ops)

            app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

            print("App data: (",app_x,app_y,")","(",app_w,app_h,")")

            saver = tf.train.Saver()

            model_file = os.getcwd() + "/backup_model/model.ckpt"

            config = tf.ConfigProto(allow_soft_placement = True)

            with tf.Session(config=config) as sess:

                init_op = tf.global_variables_initializer()
                model = sess.run(init_op)
                if os.path.isfile(os.getcwd() + "/backup_model/checkpoint"):
                    saver.restore(sess, model_file)
                    print("Restored model")
                yolo.set_training(False)

                anchors = np.reshape(np.array(cfg.anchors), [-1, 2])

                start_time = time.time()

                while (time.time() - start_time < 30):

                    image = pyautogui.screenshot().convert("L")

                    image = np.array(image)

                    image = image[app_y:app_y+app_h, app_x:app_x+app_w]

                    image = cv2.resize(image, (cfg.width, cfg.height))


                    images = np.reshape(image, [1, cfg.width, cfg.height, 1])

                    boxes = sess.run(yolo.output, feed_dict={
                        yolo.x: images,
                        yolo.anchors: anchors,
                    })

                    proc_boxes = yolo.convert_net_to_bb(boxes, filter_top=False)

                    for box_num in range(10):

                        highest_conf = proc_boxes[0][5]
                        best_box = proc_boxes[0]
                        for b in proc_boxes:
                            if (b[5] > highest_conf):
                                highest_conf = b[5]
                                best_box = b

                        x = max(10, min(app_x + app_w - 10, app_x + (best_box[1]*app_w)))
                        y = max(10, min(app_y + app_h - 10, app_y + (best_box[2]*app_h)))

                        print("Clicking", "(", x, y, ")")

                        # handle widget type:
                        widget = yolo.names[int(best_box[0])]

                        print("interacting with", widget)
                        print("BBox:", best_box)
                        if widget == "button" or widget == "combo_box" or widget == "list" or widget == "tree" or \
                                widget == "scroll_bar" or widget == "tabs" or widget == "menu" or widget == "menu_item":
                            pyautogui.click(x, y)
                        elif widget == "text_field":
                            pyautogui.click(x, y)
                            pyautogui.typewrite('Hello world!', interval=0.01)
                        else:
                            print(widget, "unrecognised")


                        del(best_box)


