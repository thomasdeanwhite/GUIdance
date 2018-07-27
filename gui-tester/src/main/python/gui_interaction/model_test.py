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
import time

def get_window_size(window_name):
    try:
        display = Xlib.display.Display()
        root = display.screen().root

        win_names = window_name.split(":")

        windowIDs = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value
        wid = 0
        win = None
        for windowID in windowIDs:
            window = display.create_resource_object('window', windowID)
            name = window.get_wm_name() # Title
            tags = window.get_wm_class()
            if tags != None and len(tags) > 1:
                name = tags[1]
            print(window.get_wm_class())
            if isinstance(name, str):
                for w_n in win_names:
                    if w_n in name:
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
            print('Screen cap failed: '+ str(e))
        return app_x, app_y, app_w, app_h
    except Exception as e:
        print('Screen cap failed: '+ str(e))
    return 0, 0, 0, 0

def generate_input_string():
    return "Hello World!"

if __name__ == '__main__':

    if len(sys.argv) > 1:
        cfg.window_name = sys.argv[1]

    with tf.device(cfg.gpu):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            event = 0

            yolo = Yolo()

            yolo.create_network()

            yolo.set_training(True)
            yolo.set_update_ops(update_ops)

            saver = tf.train.Saver()

            model_file = os.getcwd() + "/" + cfg.weights_dir + "/model.ckpt"

            config = tf.ConfigProto(allow_soft_placement = True)

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

            with tf.Session(config=config) as sess:

                init_op = tf.global_variables_initializer()
                model = sess.run(init_op)
                if os.path.isfile(os.getcwd() + "/" + cfg.weights_dir + "/checkpoint"):
                    saver.restore(sess, model_file)
                    print("Restored model")
                yolo.set_training(False)

                anchors = np.reshape(np.array(cfg.anchors), [-1, 2])

                start_time = time.time()

                last_image = None

                boxes = []

                interactions = []

                while (time.time() - start_time < 120):

                    app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)

                    while app_w == 0:
                        time.sleep(1)
                        app_x, app_y, app_w, app_h = get_window_size(cfg.window_name)
                        if time.time() - start_time > 120:
                            print("Couldn't find application window!")
                            break

                    if time.time() - start_time > 120:
                        break

                    image = pyautogui.screenshot().convert("L")

                    image = np.array(image)

                    raw_image = image[app_y:app_y+app_h, app_x:app_x+app_w]

                    image = cv2.resize(raw_image, (cfg.width, cfg.height))

                    images = np.reshape(image, [1, cfg.width, cfg.height, 1])

                    imgs = (images/127.5)-1
                    gen_boxes = True

                    for l in states:
                        diff = np.sum(np.square(image-l[0]))/image.size
                        if diff < 2:
                            gen_boxes = False
                            proc_boxes = l[1]

                    last_image = image

                    if gen_boxes or len(proc_boxes) < 3:

                        print("New state found!", len(states), "states found total.")

                        boxes = sess.run(yolo.output, feed_dict={
                            yolo.x: imgs,
                            yolo.anchors: anchors,
                        })

                        p_boxes = yolo.convert_net_to_bb(boxes, filter_top=False)

                        total = np.sum(p_boxes[:,5])

                        print(total)

                        p_boxes[:,5] = p_boxes[:,5]/total

                        print(np.sum(p_boxes[:,5]))

                        proc_boxes = p_boxes.tolist()

                        states.append([image, proc_boxes])

                    for box_num in range(1):

                        input_string = generate_input_string()

                        #highest_conf = proc_boxes[0][5]
                        best_box = proc_boxes[0]

                        rand_num = random.random()

                        for b in proc_boxes:
                            #if (b[5] > highest_conf):
                            rand_num -= b[5]
                            if rand_num <= 0:
                                #highest_conf = b[5]
                                best_box = b
                                break;

                        print(best_box)

                        height, width = raw_image.shape

                        x = int(max(app_x, min(app_x + app_w - 10, app_x + (best_box[1]*app_w))))
                        y = int(max(app_y, min(app_y + app_h - 10, app_y + (best_box[2]*app_h))))

                        y_start = max(0, min(height, int(height*(best_box[2] - best_box[4]/2))-10))
                        y_end = max(0, min(height, int(height*(best_box[2]+best_box[4]/2))+10))

                        x_start = max(0, min(width, int(width*(best_box[1]-best_box[3]/2))-10))
                        x_end = max(0, min(width, int(width*(best_box[1]+best_box[3]/2))+10))
                        image_clicked = raw_image[y_start:y_end,
                                                  x_start:x_end]

                        output_img = output_dir + "/images/" + str(event)

                        np.save(output_img, image_clicked)

                        cv2.imwrite(output_img + ".jpg", image_clicked)

                        widget = yolo.names[int(best_box[0])]

                        interactions.append([widget, event, input_string])


                        with open(test_file, "a+") as t_f:
                            t_f.write(widget + "," + str(event) + "\n")

                        event = event + 1

                        if widget == "button" or widget == "tabs" or widget == "menu" \
                                or widget == "menu_item" or widget == "toggle_button":
                            pyautogui.click(x, y)
                        elif widget == "list" or widget == "scroll_bar":
                            x = x_start + random.randint(x_end-x_start)
                            y = y_start + random.randint(y_end-y_start)
                            pyautogui.click(x_start + random.randint(x_end-x_start),
                                            y_start + random.randint(y_end-y_start))
                        elif widget == "tree":
                            x = x_start + random.randint(x_end-x_start)
                            y = y_start + random.randint(y_end-y_start)

                            pyautogui.doubleClick(x, y)

                            x = x_start + random.randint(x_end-x_start)
                            y = y_start + random.randint(y_end-y_start)

                            pyautogui.click(x, y)
                        elif widget == "text_field":
                            pyautogui.click(x, y)
                            pyautogui.typewrite(input_string, interval=0.01)
                        elif widget == "combo_box":
                            pyautogui.click(x, y)
                            event = event + 1
                            next_y = best_box[2]+random.random()*0.5
                            x = int(max(app_x, min(app_x + app_w - 10, app_x + (best_box[1]*app_w))))
                            y = int(max(app_y, min(app_y + app_h - 10, app_y + ((next_y)*app_h))))

                            y_start = max(0, min(height, int(height*(next_y - best_box[4]/2))-10))
                            y_end = max(0, min(height, int(height*(next_y+best_box[4]/2))+10))

                            x_start = max(0, min(width, int(width*(best_box[1]-best_box[3]/2))-10))
                            x_end = max(0, min(width, int(width*(best_box[1]+best_box[3]/2))+10))
                            image_clicked = raw_image[y_start:y_end,
                                            x_start:x_end]

                            output_img = output_dir + "/images/" + str(event)

                            np.save(output_img, image_clicked)

                            cv2.imwrite(output_img + ".jpg", image_clicked)
                        else:
                            print(widget, "unrecognised")

                        #proc_boxes.remove(best_box)

                print("Writing concrete tests")

                # write python regression test
                python_file = output_dir + "/test.py"

                # write visual test
                html_file = output_dir + "/test.html"

                with open(python_file, "w+") as p_f:
                    # TODO: Write python functions for click, type, etc
                    p_f.write("")

                with open(html_file, "w+") as h_f:
                    h_f.write("<html><head><title>Test " + str(runtime) + "</title></head>\n"
                                "<body>")

                for i in interactions:
                    with open(python_file, "a+") as p_f:
                        # write python regression test
                        p_f.write("click(" + str(i[1]) + ")\n")
                        if i[0] == "text_field":
                            p_f.write("type(" + i[2] + ")\n")

                    with open(html_file, "a+") as h_f:
                        # write python regression test
                        output_img = output_dir + "/images/" + str(i[1]) + ".jpg"
                        h_f.write("<br />click( <img src='" + output_img + "' /> )<br />\n")
                        if i[0] == "text_field":
                            h_f.write("type(" + i[2] + ")<br />\n")


                with open(html_file, "a+") as h_f:
                    h_f.write("</body></html>")
