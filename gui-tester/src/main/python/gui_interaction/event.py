from enum import Enum
import pyautogui

class EventType(Enum):
    NONE=-1
    LEFT_CLICK=0
    RIGHT_CLICK=1
    MOUSE_WHEEL=2
    KEY_PRESS=3


class Event():
    timestamp=0
    event_type = EventType.NONE

    def __init__(self, timestamp, event_type):
        self.timestamp = timestamp
        self.event_type = event_type

    def get_metadata(self):
        return None

    def get_event_type(self):
        return self.event_type

    def hashcode(self):
        return format("%s::%s", self.event_type.name, self.get_metadata())

    def should_perform_event(self, timestamp):
        return self.timestamp <= timestamp

    def perform(self):
        raise NotImplementedError()


class MouseEvent():
    point = (0, 0)
    def __init__(self, timestamp, mouse_button, x, y):
        super().__init__(timestamp, EventType.LEFT_CLICK if mouse_button == 0 else EventType.RIGHT_CLICK)
        self.point = (x, y)

    def get_metadata(self):
        return format("<<%d,%d>>", self.x, self.y)

    def perform(self):
        x, y = self.point
        if self.get_event_type == EventType.LEFT_CLICK:
            pyautogui.click(x, y)
        else:
            pyautogui.rightClick(x, y)

class KeyEvent():
    keys = ""
    def __init__(self, timestamp, keys):
        super().__init__(timestamp, EventType.KEY_PRESS)
        self.keys = keys

    def get_metadata(self):
        return format("<<\"%d\">>", self.keys)

    def perform(self):
        pyautogui.typewrite(self.keys, interval=0.01)

class WheelEvent():
    velocity = 0
    def __init__(self, timestamp, velocity):
        super().__init__(timestamp, EventType.KEY_PRESS)
        self.velocity = velocity

    def get_metadata(self):
        return format("<<%d>>", self.velocity)

    def perform(self):
        pyautogui.scroll(self.velocity)




