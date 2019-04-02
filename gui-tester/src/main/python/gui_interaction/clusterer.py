
import random
import numpy as np
import math

class Clusterer():

    def __init__(self):
        self.data = []
        self.shape = [0, 0]

    def append_data(self, row):
        self.data.append(row)
        if self.shape[0] == 0:
            self.shape[1] = len(row)
        self.shape[0] += 1

    def clear_data(self):
        self.shape = [0, 0]
        del self.data
        self.data = []

    def recommend_clusters(self):
        return int(math.ceil(self.shape[0]/2))

    def cluster(self, k, iterations):
        centroids = random.sample(self.data, k)

        data = np.tile(np.expand_dims(np.array(self.data), 0), [k, 1, 1])

        i = iterations

        min = []

        while i > 0:
            assignments = []

            cent = np.expand_dims(np.array(centroids), 1)

            cent = np.tile(
                np.reshape(cent, [k, 1, self.shape[1]]), [1, self.shape[0], 1])

            diff = np.sum(np.square(cent - data), axis=-1)

            min =  np.argmin(diff, axis=0).tolist()

            for i in range(k):
                assignments.append([])

            for j in range(len(min)):
                assignments[min[j]].append(self.data[j])

            new_centroids = []

            for i in range(len(assignments)):
                row = assignments[i]

                if len(row) == 0:
                    row = [[0 for x in range(self.shape[1])]]
                new_c = np.mean(row, axis=0).tolist()
                new_centroids.append(new_c)

            new_cent = False

            for c in new_centroids:
                if not c in centroids:
                    new_cent = True
            if not new_cent:
                break

            centroids = new_centroids

            i -= 1

        return centroids, min



