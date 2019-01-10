import config as cfg
from yolo import Yolo
import cv2
import numpy as np
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
import time
from pynput import keyboard
import data_loader
import signal
import timeit
import signal
from test_helper import get_window_size, screenshot


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
            print("[Swing Test] Killing tester.")

sub_window = False

error_count = -1

def kill_old_process():
    global process_id

    if process_id != -1:
        os.killpg(os.getpgid(process_id.pid), signal.SIGTERM)
        process_id = -1


def handler(signum, frame):
    print('Signal handler called with signal', signum)

    kill_old_process()

    raise TimeoutError("Could not retrieve bounding boxes!")

signal.signal(signal.SIGALRM,handler)

def start_aut():
    global working_dir, aut_command, process_id

    if aut_command != "":

        print("[Swing Test] Starting AUT")

        kill_old_process()

        os.chdir(working_dir)

        with open('stdout.log', "a+") as outfile:
            with open('stderr.log', "a+") as errfile:
                outfile.write("\n" + str(time.time()) + "\n")
                cmds = aut_command.split()
                print(cmds)
                process_id = subprocess.Popen(aut_command.split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=errfile, preexec_fn=os.setsid, shell=False,
                                              bufsize=1,
                                              universal_newlines=True)
        time.sleep(10)
    else:
        print("[Swing Test] Could not find AUT to start!")


def generate_input_string():
    if random.random() < 0.5:
        return "Hello World!"
    else:
        return str(random.randint(-10000, 10000))

def convert_coords(x, y, w, h, aspect):
    if aspect > 1: # width is bigger than height
        h = h * aspect
        y = 0.5 + ((y - 0.5)*aspect)
    elif aspect < 1:
        w = w / aspect
        x = 0.5 + ((x - 0.5)/aspect)

    return x, y, w, h

def perform_interaction(best_box):
    x = int(max(app_x+5, min(app_x + app_w - 5, app_x + (best_box[1]*app_w))))
    y = int(max(app_y+25, min(app_y + app_h - 5, app_y + (best_box[2]*app_h))))

    random_interaction = random.random()

    if random_interaction < 0.888888888888: # just a normal click
        if random.random() < 0.8:
            pyautogui.click(x, y)
        else:
            pyautogui.rightClick(x, y)
    else: # click and type 'Hello world!'
        pyautogui.click(x, y)
        pyautogui.typewrite(generate_input_string(), interval=0.01)

def select_random_box(proc_boxes):
    best_box = random.sample(proc_boxes.tolist(), 1)[0]

    return best_box

def prepare_screenshot(raw_image):
    st = time.time()
    raw_image = data_loader.pad_image(raw_image)
    et = time.time()
    if debug:
        print("[Swing Test] PADDING:", et - st)
    image = cv2.resize(raw_image, (cfg.width, cfg.height))

    images = np.reshape(image, [1, cfg.width, cfg.height, 1])

    imgs = (images/127.5)-1

    return imgs

def convert_boxes(boxes):
    return yolo.convert_net_to_bb(boxes, filter_top=False)[0]

if __name__ == '__main__':
    # Collect events until released
    with keyboard.Listener(on_release=on_release) as listener:
        if len(sys.argv) > 1:
            cfg.window_name = sys.argv[1]

        event = 0

        yolo = Yolo()

        states = []

        runtime = round(time.time())

        output_dir = cfg.output_dir + "/" + str(runtime)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(output_dir + "/images"):
            os.makedirs(output_dir + "/images")

        test_file = output_dir + "/test.txt"

        with open(test_file, "w+") as t_f:
            t_f.write("")


        if len(sys.argv) > 3:
            working_dir = sys.argv[2]
            aut_command = sys.argv[3]

            start_aut()

        yolo.set_training(False)

        anchors = np.reshape(np.array(cfg.anchors), [-1, 2])

        start_time = time.time()

        last_image = None

        boxes = []

        interactions = []

        runtime = cfg.test_time #5 mins

        last_u_input = ""

        actions = 0

        csv_file = cfg.log_dir + "/" + str(start_time) + "-test.csv"

        with open(csv_file, "w+") as p_f:
            p_f.write("time,actions,technique,iteration_time,window_name\n")

        while ((time.time() - start_time < runtime and not cfg.use_iterations) or
               (actions < cfg.test_iterations and cfg.use_iterations)) and running:

            signal.alarm(60)

            iteration_time = time.time()

            time.sleep(1)

            exec_time = time.time() - start_time

            os.system('killall "firefox"')

            app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

            counter = 0

            while app_w == 0:

                start_aut()
                counter += 1
                app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)
                if counter >= 3:
                    print("[Swing Test] Couldn't find application window!")
                    break

            if time.time() - start_time > runtime:
                break

            image = screenshot()[app_y:app_y+app_h, app_x:app_x+app_w]

            image = np.array(image)

            # raw_image = image[app_y:app_y+app_h, app_x:app_x+an.write(b"bounds")pp_w]

            ih, iw = image.shape[:2]

            aspect = iw/ih

            imgs = prepare_screenshot(image)

            gen_boxes = True

            for l in states:
                diff = np.sum(np.square(image-l[0]))/image.size
                if diff < 2:
                    gen_boxes = False
                    proc_boxes = l[1]

            last_image = image

            if True: #gen_boxes or len(proc_boxes) < 3:

                #print("New state found!", len(states), "states found total.")

                st = time.time()

                process_id.stdin.write("bounds\n")

                line = "first line"

                p_boxes = []

                while len(line) > 1:
                    line = process_id.stdout.readline()

                    if "AUT Not Supported!" in line:
                        print(line)
                        kill_old_process()
                        sys.exit(-1)

                    l_arr = line.split(",")

                    if len(l_arr) > 4:
                        error_count = 0
                        p_boxes.append([yolo.names.index(l_arr[0]),
                                        float(l_arr[1]),
                                        float(l_arr[2]),
                                        float(l_arr[3]),
                                        float(l_arr[4])])

                #p_boxes = np.array(p_boxes)

                if len(p_boxes) == 0:
                    error_count += 1

                    if error_count > 5:
                        kill_old_process()
                        print("5 errors in a row. Is application supported?")
                        sys.exit(1)

                    continue

                proc_boxes = np.array(p_boxes)

            for box_num in range(1):

                input_string = generate_input_string()

                if (process_id.poll() != None):
                    sys.exit(-1)

                best_box = select_random_box(proc_boxes)

                rand_num = random.random()

                np.delete(proc_boxes, best_box)

                height, width = image.shape[:2]

                current_box = best_box

                perform_interaction(best_box)

                actions += 1

            end_iteration_time = time.time()
            if debug:
                print("[Swing Test] Iteration Time:", end_iteration_time - iteration_time)

            # write test info
            with open(csv_file, "a") as p_f:
                p_f.write(str(exec_time) + "," + str(actions) + ",detection," + str(iteration_time) + "," + cfg.window_name + "\n")

        kill_old_process()

        time.sleep(20)
        print("[Swing Test] Finished testing!")


