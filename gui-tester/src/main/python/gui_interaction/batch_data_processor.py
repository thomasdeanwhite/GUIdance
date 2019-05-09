#!/usr/bin/sudo python3

import sys
import os
import time
import subprocess
from pynput import keyboard
import signal
from pynput.mouse import Listener, Button
from event import MouseEvent, KeyEvent, ScrollEvent, EventType, WindowChangeEvent, EventConstructor, Event
from clusterer import Clusterer
from user_model import WindowModel, UserModel
from ngram import Ngram
import pickle
import cv2


apps = [
        ("00", "bellmanzadeh"),
        ("02", "JabRef"),
        ("03", "Java_FireLands"),
        ("05", "ordrumbox"),
        ("11", "Dietetics"),
        ("12", "Minesweeper"),
        ("13", "SQuiz"),
        ("14", "BlackJack"),
        ("15", "UPM"),
        ("16", "Simple_Calculator")]

sub_window = False

normalise = True

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

windows = []

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


def gen_window_model(window_event, proc_events, clusterer=Clusterer()):
    global default_window_event
    if window_event == default_window_event.wm_name:
        return None

    assignments = {}
    clustered_events = {}

    for et in proc_events[window_event]:
        clusterer.clear_data()

        if et is EventType.NONE:
            continue

        try:
            for e in proc_events[window_event][et]:
                f = e.get_features()
                if len(f) == 0:
                    break
                clusterer.append_data(f)
            if clusterer.shape[1] == 0:
                continue
            centroids, assigns = clusterer.cluster(clusterer.recommend_clusters(), 2000)

            clustered_events[str(et)] = centroids

            for i in range(len(proc_events[window_event][et])):
                assignments[proc_events[window_event][et][i]] = assigns[i]

        except NotImplementedError as e:
            print(e)
            pass

    ngram = Ngram("")

    clustered_windowed_events = windowed_events[window_event][:]

    for i in range(len(clustered_windowed_events)):
        we = clustered_windowed_events[i]
        name = str(we.event_type)

        id = we.get_identifier()
        if not id is None:
            name += "[" + id + "]"

        if we in assignments:
            assignment = "{" + str(assignments[we]) + "}"
            if "{cluster}" in name:
                name = name.replace("{cluster}", assignment)
            else:
                name += "[" + assignment + "]"

        clustered_windowed_events[i] = name

    sequence = " ".join(clustered_windowed_events).replace("EventType.NONE", ngram.delimiter)
    ngram.construct(sequence, 5)

    ngram.calculate_probabilities()

    window_model = WindowModel(ngram, clustered_events)

    return window_model

def find(name, path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if name in f:
                return os.path.join(root, f)

if __name__ == '__main__':
    # Collect events until released

    wd = os.getcwd()

    for application in apps:
        window_event = default_window_event

        if len(sys.argv) < 2:
            print("Must have filename argument!", file=sys.stderr)
            sys.exit(1)
        input_dir = sys.argv[1]

        input_dirs = ""

        for i in range(1, 11):
            p_dir = "/participant-" + str(i)
            input_dirs += input_dir + p_dir + "/tasks/" + application[1] + ";"
            input_dirs += input_dir + p_dir + "/warmup/" + application[1] + ";"

        input_dirs = input_dirs[:-1]
        input_dirs = input_dirs.split(";")

        output_dir = input_dirs[0]

        if len(sys.argv) > 2:
            output_dir = sys.argv[2] + "/" + application[1]

        event_constructor = EventConstructor()

        for f in input_dirs:
            file = f + "/" + "events.evnt"

            if not os.path.isfile(file):
                print("Cannot find file: " + file)
                continue



            with open(file, "r") as f:
                for line in f:
                    events.append(event_constructor.construct(line))
            events.append(Event(events[-1].timestamp+1, EventType.NONE))

        events = sorted(events, key=lambda x : x.timestamp)

        proc_events = {}

        windowed_events = {}

        for e in events:

            key = window_event.filename_string()

            if isinstance(e, WindowChangeEvent):
                window_event = e
                windows.append(e)

            if not key in proc_events:
                proc_events[key] = {}

            if not e.event_type in proc_events[key]:
                proc_events[key][e.event_type] = []

            if not None in proc_events:
                proc_events[None] = {}

            if not e.event_type in proc_events[None]:
                proc_events[None][e.event_type] = []


            if normalise and not window_event == default_window_event:
                    e.change_window(window_event)

            proc_events[key][e.event_type].append(e)
            proc_events[None][e.event_type].append(e)

            if not isinstance(e, WindowChangeEvent) or window_event.filename_string() != key:

                if not None in windowed_events:
                    windowed_events[None] = []

                windowed_events[None].append(e)

                if not key in windowed_events:
                    windowed_events[key] = []

                windowed_events[key].append(e)

        user_model = UserModel()

        for window_event in proc_events:

            window_model = gen_window_model(window_event, proc_events)

            if window_model is None:
                continue

            user_model.add_window_model(window_event, window_model)

            if not window_event is None:

                screenshot = find(window_event, input_dirs[0] + "/images/")

                cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0)]
                print(screenshot)

                if not screenshot is None and os.path.isfile(screenshot):
                    img = cv2.imread(screenshot)
                    if not img is None:
                        ci = 0
                        h, w = img.shape[:2]
                        for c in window_model.clusters:
                            if c[-4:] == "DOWN":
                                for cl in window_model.clusters[c]:
                                    print(cl)
                                    cv2.rectangle(img,
                                                  (int(cl[0]*w)-2, int(cl[1]*h)-2),
                                                  (int(cl[0]*w)+2, int(cl[1]*h)+2),
                                                  cols[ci], -1, 8)
                                ci += 1
                        imgout = output_dir + "/imgs/"

                        if not os.path.isdir(imgout):
                            os.makedirs(imgout, exist_ok=True)
                            try:
                                os.mkdir(imgout)
                            except OSError as e:
                                #folder exists
                                pass


                        cv2.imwrite(imgout + window_event + ".png", img)

        default_model = gen_window_model(None, proc_events)

        user_model.set_default_model(default_model)

        for i in range(100):
            print("Gen:", default_model.next())

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            try:
                os.mkdir(output_dir)
            except OSError as e:
                #folder exists
                pass

        with open(output_dir + "/user_model.mdl", "bw+") as f:
            pickle.dump(user_model, f)









