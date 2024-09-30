import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

class KMeans():
    def __init__(self, data, k, initial_centers=None, method='Random'):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
        self.current_step = 0
        if initial_centers is not None:
            self.centers = initial_centers
        else:
            self.initialize(method)

    def initialize(self, method='Random'):
        if method == 'KMeans++':
            self.centers = self.kmeans_plus_plus()
        elif method == 'Farthest':
            self.centers = self.farthest_first()
        elif method == 'Random':
            self.centers = self.random_initialize()
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def random_initialize(self):
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def kmeans_plus_plus(self):
        centers = []
        first_center = self.data[np.random.choice(self.data.shape[0])]
        centers.append(first_center)
        for _ in range(1, self.k):
            distances = np.min([np.sum((self.data - center) ** 2, axis=1) for center in centers], axis=0)
            probabilities = distances / distances.sum()
            next_center = self.data[np.random.choice(self.data.shape[0], p=probabilities)]
            centers.append(next_center)
        return np.array(centers)

    def farthest_first(self):
        centers = [self.data[np.random.choice(self.data.shape[0])]]
        for _ in range(1, self.k):
            distances = np.min([np.linalg.norm(self.data - center, axis=1) for center in centers], axis=0)
            next_center = self.data[np.argmax(distances)]
            centers.append(next_center)
        return np.array(centers)

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            dist = float('inf')
            for j in range(self.k):
                new_dist = self.dist(centers[j], self.data[i])
                if new_dist < dist:
                    self.assignment[i] = j
                    dist = new_dist

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = []
            for j in range(len(self.assignment)):
                if self.assignment[j] == i:
                    cluster.append(self.data[j])
            if len(cluster) > 0:
                centers.append(np.mean(np.array(cluster), axis=0))
            else:
                centers.append(self.centers[i])

        return np.array(centers)

    def snap(self, centers):
        TEMPFILE = "temp.png"
        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100)
        fig.savefig(TEMPFILE)
        plt.close()
        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        return not np.allclose(centers, new_centers)

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def step(self):
        if self.current_step == 0:
            self.make_clusters(self.centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        else:
            new_centers = self.compute_centers()
            if not self.are_diff(self.centers, new_centers):
                return False  # Converged
            self.unassign()
            self.centers = new_centers
            self.make_clusters(self.centers)
            self.snap(new_centers)
        self.current_step += 1
        return True

    def lloyds(self):
        centers = self.centers
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)

        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        self.centers = new_centers  # Update centers after convergence