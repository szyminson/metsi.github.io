import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import DistanceMetric

class OOO(BaseEstimator, ClassifierMixin):
    def __init__(self, metric="euclidean", sigma=3):
        self.metric = metric
        self.sigma = sigma

    def fit(self, X, y):
        # Prepare
        self.X_ = np.copy(X)
        self.y_ = np.copy(y)
        self.classes_ = np.unique(self.y_)
        self.dist = DistanceMetric.get_metric(self.metric)

        while True:
            # Calculate centroids and deviations
            self.centroids = np.array([np.mean(self.X_[self.y_==label],axis=0)
                                       for label in self.classes_])
            self.deviations = np.array([np.std(self.X_[self.y_==label],axis=0)
                                        for label in self.classes_])
            self.borderline = self.centroids + self.deviations

            # Calculate accepted class distance as k-sigma with chosen metric
            accepted_distances = self.sigma * np.squeeze(np.array([self.dist.pairwise([self.centroids[i]], [self.borderline[i]]) for i, _ in enumerate(self.classes_)]))

            # Calculate all distances
            all_distances = self.dist.pairwise(self.centroids, self.X_)

            # Select outliers
            destroyer = np.any(np.array([all_distances[i] > accepted_distances[i] for i, _ in enumerate(self.classes_)]), axis=0)

            # Reduce training set
            self.X_ = self.X_[destroyer == False]
            self.y_ = self.y_[destroyer == False]

            # Break if no more outliers
            if np.sum(destroyer) == 0:
                break
            else:
                print(np.sum(destroyer), "outliers")

    def predict(self, X):
        return self.classes_[np.argmin(self.dist.pairwise(self.centroids, X), axis=0)]
