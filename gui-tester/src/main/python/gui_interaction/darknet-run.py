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

    app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

    start_time = time.time()

    yolo = Yolo()

    os.chdir(cfg.darknet_location)

    while (time.time() - start_time < 300):
        os.system("gnome-screenshot --file=/tmp/current_screen.png")

        image = cv2.imread("/tmp/current_screen.png")

        image = image[app_y:app_y+app_h, app_x:app_x+app_w]

        cv2.imwrite("/tmp/current_screen.png", image)

        proc_boxes =[[]]

        command =  "export DISPLAY='';./darknet detector test cfg/gui.data cfg/yolo-gui.2.0.cfg" \
                " backup/yolo-gui_9000.weights /tmp/current_screen.png -thresh 0"

        boxes_string = subprocess.run(command, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')

        boxes_string_arr = boxes_string.split("\n")

        proc_boxes = []

        for line in boxes_string_arr[1:]:
            eles = line.split(" ")

            if (len(eles) == 6):
                proc_boxes.append([int(eles[0]),
                                   float(eles[1]),
                                   float(eles[2]),
                                   float(eles[3]),
                                   float(eles[4]),
                                   float(eles[5])])




        for box_num in range(20):

            highest_conf = proc_boxes[0][5]
            best_box = proc_boxes[0]
            for b in proc_boxes:
                if (b[5] > highest_conf):
                    highest_conf = b[5]
                    best_box = b
                    b[5] = 0

            x = int(max(app_x+10, min(app_x + app_w - 10, app_x + (best_box[1]*app_w))))
            y = int(max(app_y+10, min(app_y + app_h - 10, app_y + (best_box[2]*app_h))))

            # x = best_box[1]
            # y = best_box[2]

            pyautogui.moveTo(x, y)

            # handle widget type:
            widget = yolo.names[int(best_box[0])]

            print("interacting with", widget)
            print("at position:", "(", x, y, ")")


            if widget == "button" or widget == "combo_box" or widget == "list" or widget == "tree" or \
                    widget == "scroll_bar" or widget == "tabs" or widget == "menu" or widget == "menu_item":
                pyautogui.click(x, y)
            elif widget == "text_field":
                pyautogui.click(x, y)
                pyautogui.typewrite('Hello world!', interval=0.01)
            else:
                print(widget, "unrecognised")


