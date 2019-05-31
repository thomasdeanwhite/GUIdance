import Xlib
import random
import subprocess
import config as cfg
import traceback
import os
import cv2
debug = False
import time
import pyautogui
from event import MouseEvent, KeyEvent, ScrollEvent


display = Xlib.display.Display()

def generate_input_string():
    if random.random() < 0.5:
        return "Hello World!"
    else:
        return str(random.randint(-10000, 10000))

def screenshot():
    img_file = os.environ.get("OUT_DIR", "./")
    if not img_file.endswith("/"):
        img_file += "/"

    img_file += "screenshots/"

    #convert seconds since epoch to minutes


    if not os.path.isdir(img_file):
        os.makedirs(img_file)
    img_file += str(int(time.time())) + "-screenshot.png"

    os.system("import -silent -window root " + img_file)
    img = cv2.imread(img_file, 0)

    return img

def is_running(start_time, runtime, actions, running):
    return ((time.time() - start_time < runtime and not cfg.use_iterations) or
            (actions < cfg.test_iterations and cfg.use_iterations)) and running

def perform_interaction(best_box, app_x, app_y, app_w, app_h, input_string=generate_input_string()):
    x_mod = (0.5-random.random())*best_box[3]
    y_mod = (0.5-random.random())*best_box[4]
    x = int(max(app_x, min(app_x + app_w, app_x + ((best_box[1]+x_mod)*app_w))))
    y = int(max(app_y, min(app_y + app_h, app_y + ((best_box[2]+y_mod)*app_h))))

    # y_start = max(app_y, min(app_y + height - 5, app_y + int(height*(best_box[2] - best_box[4]/2))-5))
    # y_end = max(app_y+5, min(app_y + height, app_y + int(height*(best_box[2]+best_box[4]/2))+5))
    #
    # x_start = max(app_x, min(app_x + width - 5, app_x + int(width*(best_box[1]-best_box[3]/2))-5))
    # x_end = max(app_x + 5, min(app_x + width, app_x + int(width*(best_box[1]+best_box[3]/2))+5))
    #
    # image_clicked = raw_image[y_start:y_end,
    #                           x_start:x_end]
    #
    # output_img = output_dir + "/images/" + str(event)
    #
    # np.save(output_img, image_clicked)
    #
    # cv2.imwrite(output_img + ".jpg", image_clicked)
    #
    # widget = yolo.names[int(best_box[0   ])]
    #
    # interactions.append([widget, event, input_string])
    #
    #
    # with open(test_file, "a+") as t_f:
    #     t_f.write(widget + "," + str(event) + "\n")
    #
    # pyautogui.moveTo(x, y)
    #
    # event = event + 1
    #
    # if widget == "button" or widget == "tabs" or widget == "menu" \
    #         or widget == "menu_item" or widget == "toggle_button":
    #     if (random.random() < 0.5):
    #         pyautogui.click(x, y)
    #     else:
    #         pyautogui.rightClick(x, y)
    # elif widget == "list" or widget == "scroll_bar" or widget == "slider":
    #     x = random.randint(x_start, x_end)
    #     y = random.randint(y_start, y_end)
    #     pyautogui.moveTo(x, y)
    #     action = random.random()
    #     if action < 0.4:
    #         pyautogui.click(x, y)
    #     elif action < 0.5:
    #         pyautogui.rightClick(x, y)
    #     else:
    #         pyautogui.click(x, y)
    #         x = random.randint(x_start, x_end)
    #         y = random.randint(y_start, y_end)
    #         pyautogui.dragTo(x, y)
    #         pyautogui.mouseUp(x, y)
    #         pyautogui.click(x, y)
    # elif widget == "tree":
    #     pyautogui.click(x, y)
    #     x = random.randint(x_start, x_end)
    #     y = random.randint(y_start, y_end)
    #     pyautogui.moveTo(x, y)
    #
    #     if (random.random() < 0.5):
    #         pyautogui.doubleClick(x, y)
    #     else:
    #         pyautogui.rightClick(x, y)
    #
    #     x = random.randint(x_start, x_end)
    #     y = random.randint(y_start, y_end)
    #
    #     pyautogui.moveTo(x, y)
    #
    #     pyautogui.click(x, y)
    # elif widget == "text_field":
    #     if (random.random() < 0.5):
    #         pyautogui.click(x, y)
    #     else:
    #         pyautogui.rightClick(x, y)
    #     pyautogui.typewrite(input_string, interval=0.01)
    #     # if random.random() < 0.2:
    #     #     pyautogui.press('enter')
    # elif widget == "combo_box":
    #
    #     if (random.random() < 0.2): #press right button of cbox
    #         x = x_start + ((x_end-x_start)*0.85)
    #
    #     if (random.random() < 0.5):
    #         pyautogui.click(x, y)
    #     else:
    #         pyautogui.rightClick(x, y)
    #
    #     event = event + 1
    #     next_y = best_box[2]+random.random()*0.5
    #     x = int(max(app_x, min(app_x + app_w - 10, app_x + (best_box[1]*app_w))))
    #     y = int(max(app_y, min(app_y + app_h - 10, app_y + ((next_y)*app_h))))
    #
    #     pyautogui.click(x, y)
    #
    #     y_start = max(0, min(height, int(height*(next_y - best_box[4]/2))-10))
    #     y_end = max(0, min(height, int(height*(next_y+best_box[4]/2))+10))
    #
    #     x_start = max(0, min(width, int(width*(best_box[1]-best_box[3]/2))-10))
    #     x_end = max(0, min(width, int(width*(best_box[1]+best_box[3]/2))+10))
    #     image_clicked = raw_image[y_start:y_end,
    #                     x_start:x_end]
    #
    #     output_img = output_dir + "/images/" + str(event)
    #
    #     np.save(output_img, image_clicked)
    #
    #     cv2.imwrite(output_img + ".jpg", image_clicked)
    # else:
    #     print(widget, "unrecognised")
    #

    # pyautogui.mouseUp(x, y)

    events = []

    random_interaction = random.random()

    if random_interaction < 0.888888888888: # just a normal click
        if random.random() < 0.8:
            events.append(MouseEvent(time.time(), 0, x, y))
            pyautogui.click(x, y)
        else:
            events.append(MouseEvent(time.time(), 1, x, y))
    else: # click and type 'Hello world!'
        events.append(MouseEvent(time.time(), 0, x, y))
        events.append(KeyEvent(time.time(), input_string))

    for event in events:
        event.perform()

def get_window_size(window_name):
    return get_window_size_focus(window_name, focus=True)

def get_window_size_focus(window_name, focus=True):
    global true_window, display
    sub_window = False

    try:
        root = display.screen().root

        win_names = window_name.split(":")

        win_names.append("java") # java file browser

        #windowIDs = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value

        raw_windows = root.query_tree().children

        all_windows = raw_windows

        wid = 0
        win = None
        windows = []
        while len(all_windows) > 0:
            window = all_windows.pop(0)
            #window = display.create_resource_object('window', windowID)
            try: # to handle bad windows

                matched = False

                win_name = window.get_wm_name()
                add_child = True

                while not win_name is None and "focusproxy" in win_name.lower(): # workaround for java apps
                    window = window.query_tree().parent
                    win_name = window.get_wm_name()
                    add_child = False

                name = win_name
                tag = ""
                tags = window.get_wm_class()
                if tags != None and len(tags) > 1:
                    tag = tags[1]

                children = window.query_tree().children

                if (not children is None) and len(children) > 0:
                    for w_c in children:
                        if add_child:
                            all_windows.append(w_c)

                if name is None or window.get_wm_normal_hints() is None or window.get_attributes().map_state != Xlib.X.IsViewable:
                    continue

                if isinstance(name, str) or isinstance(tag, str):
                    for w_n in win_names:
                        if w_n.lower() in name.lower() or w_n.lower() in tag.lower():
                            # if wid != 0:
                            #     sub_window = True
                            #     if random.random() < 0.05:
                            #         print("Killing window")
                            #         os.system("xkill -id " + wid)
                            #wid = windowID
                            matched = True
                            win = window
                            windows.append(win)
                            break
                            # window.set_input_focus(Xlib.X.RevertToParent, Xli b.X.CurrentTime)
                            # window.configure(stack_mode=Xlib.X.Above)
                            #prop = window.get_full_property(display.intern_atom('_NET_WM_PID'), Xlib.X.AnyPropertyType)
                            #pid = prop.value[0] # PID

                if debug:
                    print("[Window]", window.get_wm_name(), window.get_wm_class())
            except Exception as s:
                print(str(s))

        if debug:
            print("--------------------")
            for window in windows:
                print("[Selected Window]", window.get_wm_name(), window.get_wm_class())

        if len(windows) > 1 and cfg.multiple_windows:
            win_sel = None
            while win_sel is None and len(windows) > 0:
                c_win = windows.pop(-1)#random.randint(0, len(windows)-1))

                win_sel = c_win

            if not win_sel is None:
                win = win_sel

        else:
            win = windows[0]

        name = win.get_wm_name() # Title

        try:
            win_activate = subprocess.Popen("xdotool search \"" + name + "\" windowactivate --sync", shell=True)
        except Exception as e:
            pass

        if focus:
            win.set_input_focus(Xlib.X.RevertToParent, Xlib.X.CurrentTime)
            win.configure(stack_mode=Xlib.X.Above)

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
            print('[Window Parent Error] Screen cap failed: '+ str(e))
            traceback.print_stack()
        if cfg.fullscreen:
            return 0, 0, cfg.resolution[0], cfg.resolution[1]
        else:
            return app_x, app_y, app_w, app_h
    except Exception as e:
        print('[Window Error] Screen cap failed: '+ str(e))
        traceback.print_stack()
    return 0, 0, 0, 0

def get_focused_window(window_name):
    global display
    wmname = ""
    wmclass = ""
    try:

        win = display.get_input_focus().focus



        wmname = win.get_wm_name()
        wmclass = win.get_wm_class()

        if "Desktop" == wmname or "Terminal" == wmname or wmclass[0] == "nautilus":
            return wmname, wmclass, 0, 0, 0, 0

        while not wmname is None and "focusproxy" in wmname.lower(): # workaround for java apps
            win = win.query_tree().parent
            wmname = win.get_wm_name()
            wmclass = win.get_wm_class()

        if wmname is None:
            if not win is None:
                p = win.query_tree().parent
                if not p is None and not p.get_wm_name() is None:
                    win = p
                    wmname = win.get_wm_name()
                    wmclass = win.get_wm_class()
                else:
                    return wmname, wmclass, 0, 0, 0, 0

            else:
                return wmname, wmclass, 0, 0, 0, 0

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
            print('[Window Parent Error] Screen cap failed: '+ str(e))
            traceback.print_stack()
        if cfg.fullscreen:
            return wmname, wmclass, 0, 0, cfg.resolution[0], cfg.resolution[1]
        else:
            return wmname, wmclass, app_x, app_y, app_w, app_h
    except Exception as e:
        #print('[Window Error] Screen cap failed: '+ str(e))
        traceback.print_stack()
    return wmname, wmclass, 0, 0, 0, 0
