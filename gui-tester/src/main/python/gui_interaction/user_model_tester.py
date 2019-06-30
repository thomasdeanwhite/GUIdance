#!/usr/bin/sudo python3

import config as cfg
import sys
import os
import time
import subprocess
from pynput import keyboard
import signal
from test_helper import get_window_size_focus, screenshot, perform_interaction, is_running, get_focused_window
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent, EventConstructor, Event
from user_model import UserModel
import pickle
import pyautogui
import numpy as np
from threading import Thread
import cv2

sub_window = False

running = True

debug = False

quit_counter = 3

working_dir = ""
aut_command = ""
process_id = -1
pyautogui.PAUSE = 0

pyautogui.FAILSAFE = False # disables the fail-safe
seeding_key = False

ignore_windows = True

out_folder = ""

def capture_screen():
    global app_x, app_y, app_w, app_h, exec_time, out_folder

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

def on_release(key):
    global running, quit_counter, seeding_key

    if key == keyboard.Key.esc and not seeding_key:
        quit_counter -= 1
        if quit_counter == 0:
            running = False
            print("[User Model Replay] Killing tester.")

def kill_old_process():
    global process_id

    if process_id != -1:
        os.killpg(os.getpgid(process_id.pid), signal.SIGTERM)
        process_id = -1

def start_aut():
    global working_dir, aut_command, process_id, delay

    if aut_command != "":

        print("[User Model Replay] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                process_id = subprocess.Popen(aut_command.split(), stdout=outfile, stderr=errfile, preexec_fn=os.setsid)

                time.sleep(10)
                delay += 10
    else:
        print("[User Model Replay] Could not find AUT to start!")

events = []

update_window = True

default_window_event = WindowChangeEvent(0, "nowindow", "('nowindow', 'nowindow')", (0,0), (0,0))

window_event = default_window_event

delay = 0

if __name__ == '__main__':
    # Collect events until released

    with keyboard.Listener(on_release=on_release) as klistener:

        wd = os.getcwd()

        event_constructor = EventConstructor()

        if len(sys.argv) < 4:
            print("Must have filename argument!", file=sys.stderr)
            sys.exit(1)
        input_dir = sys.argv[4]
        file = input_dir + "/user_model.mdl"
        user_model = UserModel()
        with open(file, "rb") as f:
            user_model = pickle.load(f)

        output_dir = input_dir + "/tests/" + str(int(time.time()))

        out_folder = output_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            try:
                os.mkdir(output_dir)
            except OSError as e:
                #folder exists
                pass

        with open(output_dir + "/test.log", "w+") as f:
            f.write("")

        cfg.window_name = sys.argv[1]

        working_dir = sys.argv[2]
        aut_command = sys.argv[3]

        start_aut()

        print("[User Model Replay] Starting")

        for event in events:
            print(event)

        start_time = time.time()

        runtime = cfg.test_time

        actions = 0

        csv_file = cfg.log_dir + "/" + str(start_time) + "-test.csv"

        with open(csv_file, "w+") as p_f:
            p_f.write("time,actions,technique,iteration_time,window_name\n")

        last_event = Event(0, EventType.NONE)

        while is_running(start_time, runtime, actions, running):

            iteration_time = time.time()

            exec_time = time.time() - start_time

            #os.system('wmctrl -c "firefox"')

            w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)

            no_focus = 0

            while app_w == 0 or w_class is None:

                time.sleep(1)

                if no_focus >= 20:
                    kill_old_process()

                    start_aut()

                w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)
                if not is_running(start_time, runtime, actions, running):
                    print("[User Model Replay] Couldn't find application window!")
                    break

                no_focus += 1

            wm_class = "(" + w_class[0] + "," + w_class[1] + ")"

            if w_name != window_event.wm_name or wm_class != window_event.wm_class:
                window_event = WindowChangeEvent(time.time(), w_name, w_class, (app_x, app_y),
                                                 (app_w, app_h))
                os.chdir(wd)
                with open(output_dir + "/test.log", "a+") as f:
                    f.write(window_event.hashcode() + "\n")

                print(window_event.hashcode())

            window_model = user_model.get_window_model(window_event.wm_name)

            if window_model == None or (cfg.fallback == "random" and window_model == user_model.default_model):

                event = event_constructor.random_event()

                event.change_window(window_event)
                seeding_key = True
                event.perform()
                seeding_key = False
                time.sleep(event.get_delay())

                t = Thread(target=capture_screen)
                t.start()

                with open(output_dir + "/test.log", "a+") as f:
                    f.write(event.hashcode() + "\n")

                actions += event.action_count(last_event)
            else:

                if ignore_windows:
                    window_model = user_model.default_model

                event_desc = window_model.next()

                if event_desc != "WINDOW_CHANGE":

                    event = event_constructor.construct(event_desc)

                    event.change_window(window_event)
                    seeding_key = True
                    event.perform()
                    seeding_key = False
                    time.sleep(event.get_delay())

                    t = Thread(target=capture_screen)
                    t.start()

                    with open(output_dir + "/test.log", "a+") as f:
                        f.write(event.hashcode() + "\n")

                    actions += event.action_count(last_event)

                    last_event = event


            os.chdir(wd)
            with open(csv_file, "a+") as p_f:
                # TODO: Write python functions for click, type, etc
                p_f.write(str(exec_time) + "," + str(actions) + ",UserModelGenerator," + str(iteration_time) + "," + cfg.window_name + "\n")

        for k in KeyEvent.keys_down:
            pyautogui.keyUp(k)

        kill_old_process()

        time.sleep(5)
        print("[User Model Replay] Finished testing! (" + str(actions) + " actions)")

        os.chdir(wd)


