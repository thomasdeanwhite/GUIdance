#!/usr/bin/sudo python3

import config as cfg
import sys
import os
import time
import subprocess
from pynput import keyboard
import signal
from test_helper import get_window_size_focus, screenshot, perform_interaction, is_running, get_focused_window
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent, EventConstructor
from user_model import UserModel
import pickle
import pyautogui

sub_window = False

running = True

debug = False

quit_counter = 3

working_dir = ""
aut_command = ""
process_id = -1

pyautogui.FAILSAFE = False # disables the fail-safe

def on_release(key):
    global running, quit_counter

    if key == keyboard.Key.f1:
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

        while is_running(start_time, runtime, actions, running):

            time.sleep(2)

            iteration_time = time.time()

            exec_time = time.time() - start_time

            #os.system('wmctrl -c "firefox"')

            w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)

            while app_w == 0:

                #kill_old_process()

                #start_aut()

                w_name, w_class, app_x, app_y, app_w, app_h = get_focused_window(cfg.window_name)
                if time.time() - start_time > runtime:
                    print("[User Model Replay] Couldn't find application window!")
                    break

            if w_name != window_event.wm_name or w_class != window_event.wm_class:
                window_event = WindowChangeEvent(time.time(), w_name, w_class, (app_x, app_y),
                                                 (app_w, app_h))

            window_model = user_model.get_window_model(window_event.wm_name)

            event_desc = window_model.next()

            if event_desc != "WINDOW_CHANGE":

                event = event_constructor.construct(event_desc)

                event.change_window(window_event)

                print(event_desc)

                event.perform()



            with open(csv_file, "a+") as p_f:
                # TODO: Write python functions for click, type, etc
                p_f.write(str(exec_time) + "," + str(actions) + ",UserModelGenerator," + str(iteration_time) + "," + cfg.window_name + "\n")
        kill_old_process()

        for k in KeyEvent.keys_down:
            pyautogui.keyUp(k)

        time.sleep(5)
        print("[User Model Replay] Finished testing!")

        os.chdir(wd)


