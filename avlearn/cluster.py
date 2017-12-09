import numpy as np
from scipy.spatial.distance import euclidean

class my_kmeans:
    centroids = []
    clusters = []
    max_iters = None
    k = None
    labels_ = None

    def __init__(self, n_clusters, max_iters=1000):
        self.max_iters = max_iters
        self.k = n_clusters

    def fit(self, X):
        self.centroids = X[np.random.choice(np.arange(len(X)), self.k), :]
        itr = 0
        while(itr < self.max_iters):
            # Assign Clusters
            self.clusters = [[] for _ in range(self.k)]
            self.labels_ = np.array([i+1 for i in range(len(X))])
            for x in range(len(X)):
                clst_ind = np.argmin([self._euclidean_distance(X[x], cntr)
                                      for cntr in self.centroids])
                self.labels_[x] = clst_ind
                self.clusters[clst_ind].append([i for i in X[x]])

            # Update Centroids
            prev_centroids = self.centroids
            for i in range(self.k):
                for j in range(len(self.clusters[i][0])):
                    self.centroids[i][j] = np.mean([x[j] for x in self.clusters[i]])

            # Did centroids remain unchanged?
            if(np.array_equal(np.array(prev_centroids), np.array(self.centroids))):
                break
            itr += 1

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        return np.argmin([self._euclidean_distance(x, c) for c in self.centroids])

    def _euclidean_distance(self, x1, x2):
        return euclidean(x1, x2)
