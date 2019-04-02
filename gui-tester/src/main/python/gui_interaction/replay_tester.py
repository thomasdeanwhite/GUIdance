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
import time
import subprocess
import Xlib
from pynput import keyboard
import signal
import timeit
from pynput.mouse import Listener, Button
from test_helper import get_window_size_focus, screenshot, perform_interaction, is_running
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent, EventConstructor

sub_window = False

pyautogui.FAILSAFE = False # disables the fail-safe

running = True

debug = False

quit_counter = 3

working_dir = ""
aut_command = ""
process_id = -1

def on_release(key):
    global running, quit_counter

    if key == keyboard.Key.f1:
        quit_counter -= 1
        if quit_counter == 0:
            running = False
            print("[Replay] Killing tester.")

def kill_old_process():
    global process_id

    if process_id != -1:
        os.killpg(os.getpgid(process_id.pid), signal.SIGTERM)
        process_id = -1

def start_aut():
    global working_dir, aut_command, process_id, delay

    if aut_command != "":

        print("[Replay] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                process_id = subprocess.Popen(aut_command.split(), stdout=outfile, stderr=errfile, preexec_fn=os.setsid)

                time.sleep(10)
                delay += 10
    else:
        print("[Replay] Could not find AUT to start!")

mouse_pos = (0, 0)

events = []

update_window = True

def mouse_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)


def mouse_click(x, y, button, pressed):
    global events
    events.append(MouseEvent(time.time(), 0 if button == Button.left else
                                          1 if button == Button.right else 2, x, y, pressed=pressed))



def mouse_scroll(x, y, dx, dy):
    global events
    if events[-1].get_event_type() != EventType.MOUSE_WHEEL:
        events.append(ScrollEvent(time.time(), dx, dy))
    else:
        events[-1].displace(x, y)

default_window_event = WindowChangeEvent(0, "nowindow", "('nowindow', 'nowindow')", (0,0), (0,0))

window_event = default_window_event

delay = 0

def run_events(x, y, time):
    global window_event, events, default_window_event, delay, update_window
    if len(events) == 0:
        print("[Replay] Test finished!")
        return False

    event = events[0]

    while event.should_perform_event(time - delay):
        if isinstance(event, WindowChangeEvent):
            window_event = event
            print("[Replay] Changed expected window! (%s)" % window_event.wm_name)
            update_window = True

        if not window_event == default_window_event:
            if isinstance(event, MouseEvent):
                dx = x-window_event.position[0]
                dy = y-window_event.position[1]
                event.displace(dx, dy)
            elif isinstance(event, ScrollEvent):
                dx = x-window_event.position[0]
                dy = y-window_event.position[1]
                event.displace(0, 0, dx, dy)

            event.perform()
        events.pop(0)
        event = events[0]
    return True

if __name__ == '__main__':
    # Collect events until released
    wd = os.getcwd()

    if len(sys.argv) < 4:
        print("Must have filename argument!", file=sys.stderr)
        sys.exit(1)
    input_dir = sys.argv[4]
    file = input_dir + "/" + "events.evnt"

    event_constructor = EventConstructor()

    with open(file, "r") as f:
        for line in f:
            events.append(event_constructor.construct(line))

    events = sorted(events, key=lambda x : x.timestamp)
    prev_start_time = events[0].timestamp
    for i in range(len(events)):
        events[i].timestamp -= prev_start_time

    with keyboard.Listener(on_release=on_release) as klistener:
        cfg.window_name = sys.argv[1]

        working_dir = sys.argv[2]
        aut_command = sys.argv[3]

        start_aut()

        print("[Replay] Starting")

        for event in events:
            print(event)

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

            app_x, app_y, app_w, app_h = get_window_size_focus(window_event.get_title(), focus=False)

            while app_w == 0:
                exec_time = time.time() - start_time

                if not window_event == default_window_event:
                    print("[Replay] Couldn't find window", window_event.wm_name )
                    kill_old_process()

                    start_aut()

                    title = window_event.get_title()

                    app_x, app_y, app_w, app_h = get_window_size_focus(title, focus=False)

                    update_window = False

                if not run_events(app_x, app_y, exec_time):
                    break

                if not is_running(start_time, runtime+delay, actions, running):
                    print("[Replay] Couldn't find application window!")
                    break


            if not is_running(start_time, runtime+delay, actions, running):
                break

            if not run_events(app_x, app_y, exec_time):
                break

            with open(csv_file, "a+") as p_f:
                # TODO: Write python functions for click, type, etc
                p_f.write(str(exec_time) + "," + str(actions) + ",Replay," + str(iteration_time) + "," + cfg.window_name + "\n")
        kill_old_process()

        time.sleep(5)
        print("[Replay] Finished testing!")

        os.chdir(wd)


