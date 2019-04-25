from enum import Enum
import pyautogui
import re as regex

class EventType(Enum):
    NONE=-1
    LEFT_CLICK=0
    RIGHT_CLICK=1
    MIDDLE_CLICK=2
    MOUSE_WHEEL=3
    KEY_PRESS=4
    WINDOW_CHANGE=5
    LEFT_DOWN=6
    LEFT_UP=7
    RIGHT_DOWN=8
    RIGHT_UP=9
    MIDDLE_DOWN=10
    MIDDLE_UP=11


class Event():
    timestamp=0
    event_type = EventType.NONE
    window = None

    def __init__(self, timestamp, event_type):
        self.timestamp = timestamp
        self.event_type = event_type
        self.window = None

    def get_metadata(self):
        return None

    def get_event_type(self):
        return self.event_type

    def hashcode(self):
        return "{}@{}::{}".format(self.event_type.name, self.timestamp, self.get_metadata())

    def should_perform_event(self, timestamp):
        return self.timestamp <= timestamp

    def perform(self):
        pass
        #raise NotImplementedError()

    def get_features(self):
        raise NotImplementedError()

    def set_features(self, features):
        raise NotImplementedError()

    def change_window(self, window_change):
        pass

    def get_identifier(self):
        return None




class MouseEvent(Event):
    point = (0, 0)
    pressed = False
    window = None
    def __init__(self, timestamp, mouse_button, x, y, pressed=False):
        super().__init__(timestamp, (EventType.LEFT_DOWN if pressed else EventType.LEFT_UP) if mouse_button == 0 else
                                    (EventType.RIGHT_DOWN if pressed else EventType.RIGHT_UP) if mouse_button == 1 else
                                    EventType.MIDDLE_CLICK)
        self.point = (x, y)
        self.pressed = pressed

    def get_metadata(self):
        x, y = self.point

        # if not self.window == None:
        #     x, y = ((self.point[0]-self.window.position[0])/self.window.dimension[0],
        #             (self.point[1]-self.window.position[1])/self.window.dimension[1])
        return "<<{},{},\"{}\">>".format(x, y, "pressed" if self.pressed else "released")

    def perform(self):
        x, y = self.point

        if not self.window == None:
            x, y = ((self.point[0]*self.window.dimension[0])+self.window.position[0],
                    (self.point[1]*self.window.dimension[1])+self.window.position[1])

        if self.get_event_type() == EventType.LEFT_DOWN:
            pyautogui.mouseDown(x, y)
        elif self.get_event_type() == EventType.LEFT_UP:
            pyautogui.mouseUp(x, y)
        elif self.get_event_type() == EventType.RIGHT_DOWN:
            pyautogui.mouseDown(x, y, "right")
        elif self.get_event_type() == EventType.RIGHT_UP:
            pyautogui.mouseUp(x, y, "right")
        elif self.get_event_type() == EventType.MIDDLE_DOWN:
            pyautogui.mouseUp(x, y, "middle")
        elif self.get_event_type() == EventType.MIDDLE_UP:
            pyautogui.mouseUp(x, y, "middle")

    def get_features(self):
        if self.window == None:
            return [self.point[0], self.point[1]]
        else:
            return [(self.point[0]-self.window.position[0])/self.window.dimension[0],
                    (self.point[1]-self.window.position[1])/self.window.dimension[1]]

    def set_features(self, features):
        self.point = (features[0], features[1])

    def change_window(self, window_change):
        self.window = window_change

    def get_identifier(self):
        return "{cluster},\"" + ("pressed" if self.pressed else "released") + "\""

class KeyEvent(Event):
    keys = ""
    pressed = False
    shift_modifier=0
    ctrl_modifier=0
    alt_modifier=0
    altgr_modifier=0
    fn_modifier=0
    cmd_modifier=0
    keys_down = []

    def __init__(self, timestamp, keys, pressed=False, shift=0, ctrl=0, alt=0, altgr=0, fn=0, cmd=0):
        super().__init__(timestamp, EventType.KEY_PRESS)
        self.keys = keys
        self.pressed = pressed
        self.shift_modifier=shift
        self.ctrl_modifier=ctrl
        self.alt_modifier=alt
        self.altgr_modifier=altgr
        self.fn_modifier=fn
        self.cmd_modifier=cmd

        # if shift:
        #     self.keys = self.keys.upper()
        # else:
        #     self.keys = self.keys.lower()

    def get_metadata(self):
        return "<<\"{}\",\"{}\",{},{},{},{},{},{},>>".format(self.keys.lower(), "pressed" if self.pressed else "released",
                                                             self.shift_modifier, self.ctrl_modifier, self.alt_modifier,
                                                             self.altgr_modifier, self.fn_modifier, self.cmd_modifier)

    def perform(self):
        if self.pressed:
            self.keys_down.append(self.keys)
            pyautogui.keyDown(self.keys)
        else:
            if self.keys in self.keys_down:
                self.keys_down.remove(self.keys)
            pyautogui.keyUp(self.keys)

    def get_features(self):
        return [self.shift_modifier, self.ctrl_modifier, self.alt_modifier,
                self.altgr_modifier, self.fn_modifier, self.cmd_modifier]

    def set_features(self, features):
        pass

    def change_window(self, window_change):
        pass

    def get_identifier(self):
        return "\"" + self.keys + "\",\"" + ("pressed" if self.pressed else "released") + "\"" + ",{cluster}"

class ScrollEvent(Event):
    velocity_y = 0
    velocity_x = 0
    position = (0, 0)
    def __init__(self, timestamp, velocity_x, velocity_y, x_pos, y_pos):
        super().__init__(timestamp, EventType.MOUSE_WHEEL)
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.position = (x_pos, y_pos)

    def get_metadata(self):
        x, y = self.position[0], self.position[1]
        if not self.window == None:
            x, y = ((self.position[0]-self.window.position[0])/self.window.dimension[0],
                    (self.position[1]-self.window.position[1])/self.window.dimension[1])
        return "<<{},{},{},{}>>".format(self.velocity_x, self.velocity_y, x, y)

    def displace(self, x, y, xpos, ypos):
        self.velocity_x += x
        self.velocity_y += y
        self.position = (self.position[0]+xpos, self.position[1]+ypos)

    def perform(self):
        x, y = self.position[0], self.position[1]
        if not self.window == None:
            x, y = ((self.position[0]*self.window.dimension[0])+self.window.position[0],
                    (self.position[1]*self.window.dimension[1])+self.window.position[1])

        pyautogui.hscroll(self.velocity_x, x=x, y=y)
        pyautogui.scroll(self.velocity_y, x=x, y=y)

    def get_features(self):
        if self.window == None:
            return [self.velocity_y, self.velocity_x,
                    self.position[0],
                    self.position[1]]
        else:
            return [self.velocity_y, self.velocity_x,
                    (self.position[0]-self.window.position[0])/self.window.dimension[0],
                    (self.position[0]-self.window.position[1])/self.window.dimension[1]]

    def set_features(self, features):
        if self.window == None:
            self.velocity_y = features[0]
            self.velocity_x = features[1]
            self.position = (features[2], features[3])
        else:
            self.velocity_y = features[0]
            self.velocity_x = features[1]
            self.position = (self.window.position[0]+features[2]*self.window.dimension[0],
                             self.window.position[1]+features[3]*self.window.dimension[1])

    def change_window(self, window_change):
        self.window = window_change

    def get_identifier(self):
        return "{cluster}"


class WindowChangeEvent(Event):
    wm_name = ""
    wm_class = ""
    position = (0, 0)
    dimension = (0, 0)
    classes = []
    regex = regex.compile("[\(|\)|\\|']")
    def __init__(self, timestamp, w_name, w_class, position, dimension):
        super().__init__(timestamp, EventType.WINDOW_CHANGE)

        if len(w_class) < 2:
            w_class = ('', '')

        self.wm_name = w_name
        self.wm_class = "(" + w_class[0] + "," + w_class[1] + ")"
        self.position = position
        self.dimension = dimension
        self.classes = self.wm_class
        while not self.regex.search(self.classes) is None:
            self.classes = self.regex.sub("", self.wm_class)
        self.classes = self.classes.split(",")
        for c in range(len(self.classes)):
            self.classes[c] = self.classes[c].strip()

    def get_metadata(self):
        return "<<\"{}\",\"{}\",{},{},{},{}>>".format(self.wm_name, self.wm_class,
                                                      self.position[0], self.position[1],
                                                      self.dimension[0], self.dimension[1])

    def perform(self):
        pass

    def get_title(self):
        return self.wm_name + ":" + self.classes[1]

    def filename_string(self):
        return self.wm_name.replace("/", "_")

    def get_features(self):
        return []

    def set_features(self, features):
        return []

    def change_window(self, window_change):
        pass

class EventConstructor():

    meta_regex = regex.compile("[,]*[<|>]+")
    quote_regex = regex.compile("(\"[^\",]+)[,]+([^\"]+\")")

    # NONE=-1
    # LEFT_CLICK=0
    # RIGHT_CLICK=1
    # MIDDLE_CLICK=2
    # MOUSE_WHEEL=3
    # KEY_PRESS=4
    # WINDOW_CHANGE=5

    def _mouse_event(self, event_type, metadata):
        return MouseEvent(0, 0 if event_type == EventType.LEFT_CLICK or
                                  event_type == EventType.LEFT_DOWN or
                                  event_type == EventType.LEFT_UP else
                            1 if event_type == EventType.RIGHT_CLICK or
                                 event_type == EventType.RIGHT_DOWN or
                                 event_type == EventType.RIGHT_UP else
                            2,
                          float(metadata[0]), float(metadata[1]), metadata[2] == "pressed")

    def _window_event(self, event_type, metadata):
        return WindowChangeEvent(0, metadata[0], metadata[1],
                                 (metadata[2], metadata[3]),
                                 (metadata[4], metadata[5]))

    def _key_event(self, event_type, metadata):
        return KeyEvent(0, metadata[0], metadata[1] == "pressed", metadata[2], metadata[3], metadata[4], metadata[5],
                        metadata[6], metadata[7])

    def _scroll_event(self, event_type, metadata):
        return ScrollEvent(0, metadata[0], metadata[1], metadata[2], metadata[3])

    def construct(self, serialized):
        data = serialized.strip().split("::")
        d = data[0].split("@")
        d.append(data[1])

        metadata = self.metadata_to_list(d[2])
        event_type = EventType[d[0]]
        timestamp = d[1]

        event = Event(0, event_type)

        constructor = {
            EventType.LEFT_CLICK: self._mouse_event,
            EventType.MIDDLE_CLICK: self._mouse_event,
            EventType.RIGHT_CLICK: self._mouse_event,
            EventType.WINDOW_CHANGE: self._window_event,
            EventType.KEY_PRESS: self._key_event,
            EventType.MOUSE_WHEEL: self._scroll_event,

            EventType.LEFT_DOWN: self._mouse_event,
            EventType.MIDDLE_DOWN: self._mouse_event,
            EventType.RIGHT_DOWN: self._mouse_event,

            EventType.LEFT_UP: self._mouse_event,
            EventType.MIDDLE_UP: self._mouse_event,
            EventType.RIGHT_UP: self._mouse_event
        }

        if event_type in constructor:
            event = constructor[event_type](event_type, metadata)

        event.timestamp = float(timestamp)
        return event

    def metadata_to_list(self, metadata):
        metadata = self.meta_regex.sub("", metadata)
        while not self.quote_regex.search(metadata) is None:
            metadata = self.quote_regex.sub("\g<1>%44\g<2>", metadata)

        data = metadata.split(",")
        for i in range(len(data)):

            data[i] = data[i].replace("%44", ",")

            data[i] = data[i].strip()

            if data[i][0] != "\"" and data[i][-1] != "\"":
                data[i] = float(data[i])
            else:
                data[i] = data[i][1:-1]

        return data


