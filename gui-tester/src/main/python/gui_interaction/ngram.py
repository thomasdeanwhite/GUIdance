class Ngram ():

    delimiter = "\n"

    def __init__(self, name):
        self.name = name
        self.children = []
        self.count = 0
        self.probability = 0
        self.length = 0

    def set_count(self, count):
        self.count = count

    def increment_count(self):
        self.count += 1

    def add_child(self, child):
        self.children.append(child)

    def get_child(self, name):
        for ng in self.children:
            if ng.name == name:
                return ng

        return None

    def find_children(self, names):
        nms = names.split() # split on space

        node = self

        for n in nms:
            node = node.get_child(n)

            if node == None:
                break

        return node

    def find_parent(self, names):
        nms = names.split() # split on space

        node = self

        for n in nms:
            node = node.get_child(n)

            if node == None:
                node = self
                break

        return node

    def construct(self, data, length):
        words = data.split()

        if length > len(words):
            length = len(words)-1

        self.length = length

        for lower_lim in range(len(words)-length):

            word_string = words[lower_lim]

            n = self.find_children(word_string)

            if n is None:
                n = Ngram(words[lower_lim])
                n.set_count(1)

                self.add_child(n)
            else:
                n.increment_count()

            node = n

            for upper_lim in range(1, length):
                word_string += " " + words[lower_lim+upper_lim]

                if self.delimiter in word_string:
                    continue

                n2 = self.find_children(word_string)

                if n2 is None:
                    n2 = Ngram(words[lower_lim+upper_lim])
                    n2.set_count(1)

                    node.add_child(n2)
                else:
                    n2.increment_count()

                node = n2

    def calculate_probabilities(self):

        total = 0

        for c in self.children:
            total += c.count

        for c in self.children:
            c.probability = c.count / total
            c.calculate_probabilities()



    def print_index(self, i):
        print(" "*i, self.name, "(", self.count, ")")

        for c in self.children:
            c.print_index(i+1)

    def print(self):
        self.print_index(0)