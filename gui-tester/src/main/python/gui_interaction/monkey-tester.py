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
            # prop = window.get_full_property(display.intern_atom('_NET_WM_PID'), Xlib.X.AnyPropertyType)
            # pid = prop.value[0] # PID
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

    app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

    start_time = time.time()

    while (time.time() - start_time < 30):

        x = int(max(app_x+10, min(app_x + app_w - 10, app_x + (random.random()*app_w))))
        y = int(max(app_y+10, min(app_y + app_h - 10, app_y + (random.random()*app_h))))

        print("Clicking", "(", x, y, ")")

        random_interaction = random.random()

        if random_interaction < 0.888888888888: # just a normal click
            pyautogui.click(x, y)
        else: # click and type 'Hello world!'
            pyautogui.click(x, y)
            pyautogui.typewrite('Hello world!', interval=0.01)


