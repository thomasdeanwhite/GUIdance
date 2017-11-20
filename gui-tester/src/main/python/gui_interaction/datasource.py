# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import tensorflow as tf

class DataSource:
    x = []
    y = []

    def __init__(self):
        self.x = []
        self.y = []

    def __add__(self, other):
        self.x = self.x.append(other.x)
        self.y = self.y.append(other.y)