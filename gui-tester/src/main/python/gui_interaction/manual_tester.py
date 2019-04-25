#!/usr/bin/sudo python3

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
from pynput import keyboard
import signal
import timeit
from pynput.mouse import Listener, Button
from test_helper import get_focused_window, screenshot, perform_interaction, is_running
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent
from threading import Thread

sub_window = False

running = True

debug = False

quit_counter = 3

working_dir = ""
aut_command = ""
process_id = -1

def on_release(key):
    global running, quit_counter, shift_modifier, ctrl_modifier, alt_modifier, altgr_modifier, fn_modifier, cmd_modifier

    if hasattr(key, "name"):
        name = key.name
    elif hasattr(key, "char"):
        name = key.char
    else:
        name = str(key)

    if key == keyboard.Key.shift:
        shift_modifier = 0
    if key == keyboard.Key.ctrl:
        ctrl_modifier = 0
    if key == keyboard.Key.alt:
        alt_modifier = 0
    if key == keyboard.Key.alt_gr:
        altgr_modifier = 0
    if key == keyboard.Key.cmd:
        cmd_modifier = 0

    add_event(KeyEvent(time.time(), name, pressed=False, shift=shift_modifier, ctrl=ctrl_modifier, alt=alt_modifier,
                           altgr=altgr_modifier, fn=fn_modifier, cmd=cmd_modifier))

    if key == keyboard.Key.esc:
        quit_counter -= 1
        if quit_counter == 0:
            running = False
            print("[Manual] Killing tester.")

wd = ""

def on_press(key):
    global shift_modifier, ctrl_modifier, alt_modifier, altgr_modifier, fn_modifier, cmd_modifier
    name = ""

    if hasattr(key, "name"):
        name = key.name
    elif hasattr(key, "char"):
        name = key.char
    else:
        name = str(key)

    if key == keyboard.Key.shift:
        shift_modifier = 1
    if key == keyboard.Key.ctrl:
        ctrl_modifier = 1
    if key == keyboard.Key.alt:
        alt_modifier = 1
    if key == keyboard.Key.alt_gr:
        altgr_modifier = 1
    if key == keyboard.Key.cmd:
        cmd_modifier = 1

    add_event(KeyEvent(time.time(), name, pressed=True, shift=shift_modifier, ctrl=ctrl_modifier, alt=alt_modifier,
                           altgr=altgr_modifier, fn=fn_modifier, cmd=cmd_modifier))

def kill_old_process():
    global process_id

    if process_id != -1:
        os.killpg(os.getpgid(process_id.pid), signal.SIGTERM)
        process_id = -1

def start_aut():
    global working_dir, aut_command, process_id, wd

    if aut_command != "":

        print("[Manual] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                process_id = subprocess.Popen(aut_command.split(), stdout=outfile, stderr=errfile, preexec_fn=os.setsid)

                time.sleep(10)
    else:
        print("[Manual] Could not find AUT to start!")

    os.chdir(wd)

mouse_pos = (0, 0)

events = []

shift_modifier=0
ctrl_modifier=0
alt_modifier=0
altgr_modifier=0
fn_modifier=0
cmd_modifier=0
out_folder = ""

w_name, w_class, app_x, app_y, app_w, app_h = "", "", 0, 0, 0, 0

def capture_screen():
    global app_x, app_y, app_w, app_h, exec_time

    time.sleep(1)

    img_folder = out_folder + "/images"

    img_out = img_folder + "/"+ window_event.filename_string()

    counter = int(exec_time/5)

    if not (os.path.isfile(img_out + str(counter) + ".png")):
        image = screenshot()

        if not image is None:

            if not cfg.fullscreen:
                image = image[app_y:app_y+app_h, app_x:app_x+app_w]

            image = np.array(image)


            if not os.path.isdir(img_folder):
                os.makedirs(img_folder, exist_ok=True)
                try:
                    os.mkdir(img_folder)
                except OSError as e:
                    #folder exists
                    pass

            if not (os.path.isfile(img_out + str(counter) + ".png")):
                cv2.imwrite(img_out + str(counter) + ".png", image)

def add_event(event):
    global events

    t = Thread(target=capture_screen)
    t.start()

    events.append(event)

def mouse_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)


def mouse_click(x, y, button, pressed):
    global events
    add_event(
        MouseEvent(time.time(), 0 if button == Button.left else
        1 if button == Button.right else 2, x, y, pressed=pressed)
    )



def mouse_scroll(x, y, dx, dy):
    global events
    if events[-1].get_event_type() != EventType.MOUSE_WHEEL or dy * events[-1].velocity_y < 0  \
        or dx * events[-1].velocity_x < 0:
        add_event(ScrollEvent(time.time(), dx, dy, x, y))
    else:
        events[-1].displace(x, y, 0, 0)

window_event = WindowChangeEvent(0, "", "", (0, 0), (0, 0))
exec_time = 0
if __name__ == '__main__':
    # Collect events until released
    wd = os.getcwd()
    with Listener(on_move=mouse_move,
                  on_click=mouse_click,
                  on_scroll=mouse_scroll) as listener:
        with keyboard.Listener(on_release=on_release,
                               on_press=on_press) as klistener:
            if len(sys.argv) > 1:
                cfg.window_name = sys.argv[1]

            if len(sys.argv) > 3:
                working_dir = sys.argv[2]
                aut_command = sys.argv[3]

                start_aut()

            if len(sys.argv) > 5:
                cfg.test_time = int(sys.argv[5])
                cfg.use_iterations = False
                print("[Manual] Using time limit: " + str(cfg.test_time))

            out_folder = sys.argv[4]

            output_file = out_folder + "/events.evnt"

            print("[Manual] Starting")

            start_time = time.time()

            runtime = cfg.test_time

            actions = 0

            csv_file = cfg.log_dir + "/" + str(start_time) + "-test.csv"

            with open(csv_file, "w+") as p_f:
                p_f.write("time,actions,technique,iteration_time,window_name\n")

            while is_running(start_time, runtime, actions, running):

                iteration_time = time.time()

                exec_time = time.time() - start_time

                #os.system('wmctrl -c "firefox"')

                w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)

                no_focus = 0

                while app_w == 0:

                    time.sleep(1)

                    if no_focus >= 3:
                        kill_old_process()

                        start_aut()

                    w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)
                    if not is_running(start_time, runtime, actions, running):
                        print("[Manual] Couldn't find application window!")
                        break

                    no_focus += 1

                if w_name != window_event.wm_name or "(" + w_class[0] + "," + w_class[1] + ")" != window_event.wm_class:
                    window_event = WindowChangeEvent(time.time(), w_name, w_class, (app_x, app_y),
                                                     (app_w, app_h))
                    add_event(window_event)

                if not is_running(start_time, runtime, actions, running):
                    break

                with open(csv_file, "a+") as p_f:
                    p_f.write(str(exec_time) + "," + str(actions) + ",manual," + str(iteration_time) + "," + cfg.window_name + "\n")
            kill_old_process()

            time.sleep(5)
            print("[Manual] Finished testing!")

            os.chdir(wd)

            if not os.path.exists(os.path.dirname(output_file)):
                try:
                    os.makedirs(os.path.dirname(output_file))
                except OSError as exc:
                    pass

            with open(output_file, "w+") as f:
                for event in events:
                    f.write(event.hashcode() + "\n")


