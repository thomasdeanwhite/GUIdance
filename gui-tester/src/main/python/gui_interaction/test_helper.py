import Xlib
import random
import subprocess
import config as cfg
import traceback
import os
import cv2
debug = False
import time


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


def get_window_size(window_name):
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

                children = window.query_tree().children

                if (not children is None) and len(children) > 0:
                    for w_c in children:
                        all_windows.append(w_c)

                name = window.get_wm_name() # Title
                tags = window.get_wm_class()
                if tags != None and len(tags) > 1:
                    name = tags[1]

                if name == None:
                    continue

                if debug:
                    print("[Window]", window.get_wm_class())

                if isinstance(name, str):
                    for w_n in win_names:
                        if w_n.lower() in name.lower():
                            # if wid != 0:
                            #     sub_window = True
                            #     if random.random() < 0.05:
                            #         print("Killing window")
                            #         os.system("xkill -id " + wid)
                            #wid = windowID
                            win = window
                            windows.append(win)
                            # window.set_input_focus(Xlib.X.RevertToParent, Xli b.X.CurrentTime)
                            # window.configure(stack_mode=Xlib.X.Above)
                            #prop = window.get_full_property(display.intern_atom('_NET_WM_PID'), Xlib.X.AnyPropertyType)
                            #pid = prop.value[0] # PID
            except:
                pass

        if len(windows) > 1 and cfg.multiple_windows:
            win = random.sample(windows, 1)[0]

        name = win.get_wm_name() # Title
        tags = win.get_wm_class()
        if tags != None and len(tags) > 1:
            name = tags[1]

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
            print('[Window] Screen cap failed: '+ str(e))
            traceback.print_stack()
        return app_x, app_y, app_w, app_h
    except Exception as e:
        print('[Window] Screen cap failed: '+ str(e))
        traceback.print_stack()
    return 0, 0, 0, 0
