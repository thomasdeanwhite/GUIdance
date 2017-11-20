# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
from gui_interaction.datasource import DataSource
import os
import sys
import tensorflow as tf

class GuiData(DataSource):

    def __init__(self, file=None, directory=None):
        if file != None && directory != None:
            wd = directory
            os.chdir(os.path.join(wd, sys.argv[i]))
            print("loading", sys.argv[i])
            with open('training_inputs.csv', 'rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                counter = len(raw_data)
                for row in reader:
                    raw_data.append([])
                    for e in row:
                        raw_data[counter].append(float(e))
                    counter += 1

            with open('training_outputs.csv', 'rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                counter = len(output)
                for row in reader:
                    output.append([])
                    for e in row:
                        output[counter].append(float(e))
                    counter += 1

    def __add__(self, other):
        self.data = self.data.append(other.data)