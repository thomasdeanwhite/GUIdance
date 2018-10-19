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


sub_window = False

running = True

debug = False

quit_counter = 3

working_dir = ""
aut_command = ""
process = -1

def on_release(key):
    global running, quit_counter
    if key == keyboard.Key.f1:
        quit_counter -= 1
        if quit_counter == 0:
            running = False
            print("[Random] Killing tester.")

def kill_old_process():
    global process

    if process != -1:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process = -1

def start_aut():
    global working_dir, aut_command, process

    if aut_command != "":
        print("[Random] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                process = subprocess.Popen(aut_command.split(), stdout=outfile, stderr=errfile)

                time.sleep(10)
    else:
        print("[Random] Could not find AUT to start!")

def get_window_size(window_name):
    global sub_window
    try:
        display = Xlib.display.Display()
        root = display.screen().root

        win_names = window_name.split(":")

        win_names.append("java") # java file browser

        windowIDs = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value
        wid = 0
        win = None
        for windowID in windowIDs:
            window = display.create_resource_object('window', windowID)
            name = window.get_wm_name() # Title
            tags = window.get_wm_class()
            if tags != None and len(tags) > 1:
                name = tags[1]
            print("[Random]", window.get_wm_class())
            if isinstance(name, str):
                for w_n in win_names:
                    if w_n.lower() in name.lower():
                        # if wid != 0:
                        #     sub_window = True
                        #     if random.random() < 0.05:
                        #         print("Killing window")
                        #         os.system("xkill -id " + wid)
                        wid = windowID
                        win = window
                        window.set_input_focus(Xlib.X.RevertToParent, Xlib.X.CurrentTime)
                        window.configure(stack_mode=Xlib.X.Above)
                        #prop = window.get_full_property(display.intern_atom('_NET_WM_PID'), Xlib.X.AnyPropertyType)
                        #pid = prop.value[0] # PID

        geom = win.get_geometry()

        app_x, app_y, app_w, app_h = (geom.x, geom.y, geom.width, geom.height)

        try:
            parent_win = win.query_tree().parent

            while parent_win != 0:
                #print(parent_win)
                p_geom = parent_win.get_geometry()
                app_x += p_geom.x
                app_y += p_geom.y
                parent_win = parent_win.query_tree().parent
        except Exception as e:
            print('[Random] Screen cap failed: '+ str(e))
        return app_x, app_y, app_w, app_h
    except Exception as e:
        print('[Random] Screen cap failed: '+ str(e))
    return 0, 0, 0, 0

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

        print("[Random] Starting in 5 seconds")

        time.sleep(5)


        start_time = time.time()

        runtime = cfg.test_time

        actions = 0

        while ((time.time() - start_time < runtime and not cfg.use_iterations) or
               (actions < cfg.test_iterations and cfg.use_iterations)) and running:

            iteration_time = time.time()

            exec_time = time.time() - start_time

            os.system('wmctrl -c "firefox"')

            app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

            while app_w == 0:

                kill_old_process()

                start_aut()

                app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)
                if time.time() - start_time > runtime:
                    print("[Random] Couldn't find application window!")
                    break

            if time.time() - start_time > runtime:
                break


            x = int(max(app_x+5, min(app_x + app_w - 5, app_x + (random.random()*app_w))))

            #app_y+25 for the title screen
            y = int(max(app_y+25, min(app_y + app_h - 5, app_y + (random.random()*app_h))))

            random_interaction = random.random()

            if random_interaction < 0.888888888888: # just a normal click
                if random.random() < 0.5:
                    pyautogui.doubleClick(x, y, interval=0.01)
                else:
                    pyautogui.rightClick(x, y)
            else: # click and type 'Hello world!'
                pyautogui.click(x, y, interval=0.01)
                pyautogui.typewrite(generate_input_string(), interval=0.01)

            end_iteration_time = time.time()
            if debug:
                print("[Random] Iteration Time:", end_iteration_time - iteration_time)

            # write test info
            csv_file = "test.csv"

            actions += 1

            with open(csv_file, "a") as p_f:
                # TODO: Write python functions for click, type, etc
                p_f.write(str(exec_time) + "," + str(actions) + ",detection," + str(iteration_time) + "," + cfg.window_name + "\n")
            # if sub_window:
            #     if random.random() < 0.05:
            #         pyautogui.press('escape')
        kill_old_process()

        time.sleep(20)
        print("[Random] Finished testing!")


