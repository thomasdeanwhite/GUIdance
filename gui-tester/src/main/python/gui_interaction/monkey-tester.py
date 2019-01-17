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
from test_helper import get_window_size, is_running

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
            print("[Random] Killing tester.")

def kill_old_process():
    global process_id

    if process_id != -1:
        os.killpg(os.getpgid(process_id.pid), signal.SIGTERM)
        process_id = -1

def start_aut():
    global working_dir, aut_command, process_id

    if aut_command != "":

        print("[Random] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                process_id = subprocess.Popen(aut_command.split(), stdout=outfile, stderr=errfile, preexec_fn=os.setsid)

                time.sleep(10)
    else:
        print("[Random] Could not find AUT to start!")


def generate_input_string():
    if random.random() < 0.5:
        return "Hello World!"
    else:
        return str(random.randint(-10000, 10000))

if __name__ == '__main__':
    # Collect events until released
    with keyboard.Listener(on_release=on_release) as listener:
        if len(sys.argv) > 1:
            cfg.window_name = sys.argv[1]

        if len(sys.argv) > 3:
            working_dir = sys.argv[2]
            aut_command = sys.argv[3]

            start_aut()

        print("[Random] Starting")

        start_time = time.time()

        runtime = cfg.test_time

        actions = 0

        csv_file = cfg.log_dir + "/" + str(start_time) + "-test.csv"

        with open(csv_file, "w+") as p_f:
            p_f.write("time,actions,technique,iteration_time,window_name\n")

        while is_running(start_time, runtime, actions, running):
            time.sleep(1)
            iteration_time = time.time()

            exec_time = time.time() - start_time

            os.system('killall "firefox"')

            app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

            counter = 0

            while app_w == 0:

                start_aut()
                counter += 1
                app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)
                if counter >= 3:
                    print("[Detection] Couldn't find application window!")
                    break

            if not is_running(start_time, runtime, actions, running):
                break


            x = int(max(app_x+5, min(app_x + app_w - 5, app_x + (random.random()*app_w))))

            #app_y+25 for the title screen
            y = int(max(app_y+25, min(app_y + app_h - 5, app_y + (random.random()*app_h))))

            random_interaction = random.random()

            if random_interaction < 0.888888888888: # just a normal click
                if random.random() < 0.8:
                    pyautogui.click(x, y)
                else:
                    pyautogui.rightClick(x, y)
            else: # click and type 'Hello world!'
                pyautogui.click(x, y)
                pyautogui.typewrite(generate_input_string(), interval=0.01)

            end_iteration_time = time.time()
            if debug:
                print("[Random] Iteration Time:", end_iteration_time - iteration_time)

            # write test info
            actions += 1

            with open(csv_file, "a+") as p_f:
                # TODO: Write python functions for click, type, etc
                p_f.write(str(exec_time) + "," + str(actions) + ",random," + str(iteration_time) + "," + cfg.window_name + "\n")
            # if sub_window:
            #     if random.random() < 0.05:
            #         pyautogui.press('escape')

        kill_old_process()

        time.sleep(20)
        print("[Random] Finished testing!")


