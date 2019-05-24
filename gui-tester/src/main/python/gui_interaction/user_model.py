from ngram import Ngram
import random
import re
import time

class UserModel():

    default_chance = 0.05

    def __init__(self):
        self.window_models = {}
        self.default_model = None

    def set_default_model(self, default_model):
        self.default_model = default_model

    def add_window_model(self, window_name, window_model):
        self.window_models[window_name] = window_model

    def get_window_model(self, window_name):
        
        model = self.default_model
        if window_name in self.window_models and random.random() > self.default_chance:
            model = self.window_models[window_name]

        if model is None:
            raise ValueError("Window " + window_name + " cannot be found, has model equal to None, or default model is None!")
        return model


class WindowModel():

    exp = "{(\d+)}"

    def __init__(self, ngram, clusters, length=2):
        self.chain = MarkovChain(ngram, length)
        self.clusters = clusters

    def label_to_event_description(self, label):
        m = re.search(self.exp, label)
        if not m is None: # has clustered information!
            c = int(m.group(1))
            nt = label.split("[")[0]
            metadata = self.clusters[nt][c]
            cluster = ""

            for i in metadata:
                cluster += str(i) + ","
            cluster = cluster[:-2]

            label = re.sub(self.exp, cluster, label)

        label = label.replace("[", "::<<")
        label = label.replace("]", ">>")
        return label

    def next(self):
        label = self.chain.next()

        if label is None:
            return None

        label = self.label_to_event_description(label)

        label = label.replace("EventType.", "")

        label = label.replace("::<<", "@" + str(time.time()) + "::<<")

        return label

class MarkovChain():
    def __init__(self, ngram, length):
        self.ngram = ngram
        self.sequence = ""
        if length > self.ngram.length:
            length = self.ngram.length
        self.length = length
        self.sparsity = 0

    def next(self):
        seqs = self.sequence.split()

        if len(seqs) >= self.length:
            seqs = seqs[-self.length+1:]

        self.sequence = " ".join(seqs)

        node = self.ngram.find_parent(self.sequence)

        p = random.random()

        if len(node.children) == 0:
            # sparse data
            if self.sparsity > 3:
                return None
            self.sparsity += 1
            self.sequence = ""
            return self.next()

        self.sparsity = 0

        next_node = node.children[0]
        for c in node.children:
            next_node = c
            p -= c.probability

            if p <= 0:
                break

        self.sequence += " " + next_node.name
        return next_node.name


