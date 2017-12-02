import numpy as np

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
        self.labels_ = np.array([i for i in range(len(X))])
        self.centroids = X[np.random.choice(np.arange(len(X)), self.k), :]
        itr = 0
        while(itr < self.max_iters):
            # Assign Clusters
            self.clusters = [[] for _ in range(self.k)]
            for x in X:
                clst_ind = np.argmin([self._square_distance(x, cntr)
                                      for cntr in self.centroids])
                self.clusters[clst_ind].extend(x)

            # Update Centroids
            prev_centroids = self.centroids
            for i in range(self.k):
                self.centroids[i][0] = np.mean(np.array(self.clusters[i][0]))
                self.centroids[i][1] = np.mean(np.array(self.clusters[i][1]))

            # Did centroids remain unchanged?
            if(np.array_equal(np.array(prev_centroids), np.array(self.centroids))):
                break
            itr += 1

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        return np.argmin([self._square_distance(x, c) for c in self.centroids])

    def _square_distance(self, x1, x2):
        _dist = np.linalg.norm(x1 - x2)
        return np.dot(_dist, _dist)
