#!/usr/bin/sudo python3

import sys
import os
import time
import subprocess
import signal
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent, EventConstructor, Event
from clusterer import Clusterer
from user_model import WindowModel, UserModel
from ngram import Ngram
import pickle
import cv2
import numpy as np


def create_img(w_m, screenshot):
    w = screenshot.shape[1]
    h = screenshot.shape[0]
    s_copy = np.copy(screenshot)

    for cl in w_m.clusters["EventType.LEFT_DOWN"]:

        cv2.rectangle(s_copy,
                      (int(cl[0]*w)-2, int(cl[1]*h)-2),
                      (int(cl[0]*w)+2, int(cl[1]*h)+2),
                      (0, 255, 0), -1, 8)

    return s_copy

if __name__ == '__main__':
    wd = os.getcwd()

    if len(sys.argv) < 4:
        print("Must have filename argument!", file=sys.stderr)
        sys.exit(1)
    input_dir = sys.argv[1]
    file = input_dir + "/user_model.mdl"
    user_model = UserModel()

    output = "."

    if len(sys.argv) > 4:
        output = sys.argv[4]

    with open(file, "rb") as f:
        user_model = pickle.load(f)

    screenshot_file = sys.argv[2]

    screenshot = cv2.imread(screenshot_file)

    window_title = sys.argv[3]

    output += "/" + window_title

    w_m = user_model.get_window_model(window_title)

    if w_m == user_model.default_model:
        print("Window model not found for title: '" + window_title + "'.")
        screen_app_model = create_img(user_model.default_model, screenshot)

        cv2.imwrite(output + "-one-out.png", screen_app_model)
    else:

        screen_app_model = create_img(user_model.default_model, screenshot)
        screen_window_model = create_img(w_m, screenshot)

        cv2.imwrite(output + "-app.png", screen_app_model)
        cv2.imwrite(output + "-window.png", screen_window_model)










