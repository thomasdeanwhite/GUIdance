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

def screenshot():
    img_file = os.environ.get("OUT_DIR", "./"+str(time.time()))
    if not img_file.endswith("/"):
        img_file += "/"

    if not os.path.isdir(img_file):
        os.makedirs(img_file)
    img_file += "screenshot.png"

    os.system("import -window root " + img_file)
    img = cv2.imread(img_file, 0)

    return img

def perform_interaction(best_box, app_x, app_y, app_w, app_h, input_string):
    x_mod = 0#(0.5-random.random())*best_box[3]
    y_mod = 0#(0.5-random.random())*best_box[4]
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

    random_interaction = random.random()

    if random_interaction < 0.888888888888: # just a normal click
        if random.random() < 0.8:
            pyautogui.click(x, y)
        else:
            pyautogui.rightClick(x, y)
    else: # click and type 'Hello world!'
        pyautogui.click(x, y)
        pyautogui.typewrite(input_string, interval=0.01)

def get_window_size(window_name):
    global true_window
    sub_window = False

    try:
        display = Xlib.display.Display()
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


                win_name = window.get_wm_name() # Title
                name = win_name
                tag = ""
                tags = window.get_wm_class()
                if tags != None and len(tags) > 1:
                    tag = tags[1]

                children = window.query_tree().children

                if (not children is None) and len(children) > 0:
                    for w_c in children:
                        all_windows.append(w_c)

                if name is None or window.get_wm_normal_hints() is None or window.get_attributes().map_state != Xlib.X.IsViewable:
                    continue

                if isinstance(name, str) or isinstance(tag, str):
                    for w_n in win_names:
                        if w_n.lower() in name.lower() or w_n.lower() in tag:
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
            except:
                pass

        if debug:
            print("--------------------")
            for window in windows:
                print("[Selected Window]", window.get_wm_name(), window.get_wm_class())

        if len(windows) > 1 and cfg.multiple_windows:
            win_sel = None
            while (win_sel is None or not win_sel.get_wm_icon_size() is None) and len(windows) > 0:
                c_win = windows.pop(random.randint(0, len(windows)-1))

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
