import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt


class OptimizedNearestCentroid(BaseEstimator, ClassifierMixin):
    """
    Nearest Centroid Classifier optimized according to the three sigma rule.
    """

    def __init__(self, metric='euclidean', optimize=True, sigma=3):
        self.metric = metric
        self.optimize = optimize
        self.sigma = sigma
        self.dm = DistanceMetric.get_metric(self.metric)

    def fit(self, X, y):
        # czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # przechowanie unikalnych klas problemu
        self.classes_ = unique_labels(y)

        self.X_, self.y_ = X, y

        # plt.scatter(self.X_[:, 0], self.X_[:, 1], c=y)

        self.centroids_ = []
        for cl in self.classes_:
            X_class = self.X_[self.y_ == cl]

            self.optimize_ = True
            while self.optimize_:
                if hasattr(self, 'outliers_'):
                    X_class = np.delete(X_class, self.outliers_, axis=0)

                class_centroid = np.sum(X_class, axis=0) / X_class.shape[0]
                distances = np.squeeze(self.dm.pairwise(
                    class_centroid.reshape(1, X_class.shape[1]), X_class))
                std = np.std(X_class)

                # plt.scatter(class_centroid[0], class_centroid[1], c='red')
                # plt.savefig("cos")

                self.outliers_ = np.argwhere(distances > self.sigma * std)
                if self.outliers_.shape[0] == 0 or self.optimize == False:
                    self.optimize_ = False

            self.centroids_.append(class_centroid)

        return self

    def predict(self, X):
        # sprawdzenie do wywolany zostal fit
        check_is_fitted(self)
        # sprawdzenie wejscia
        X = check_array(X)

        distance_pred= self.dm.pairwise(self.centroids_, X)
        y_pred = np.argmin(distance_pred, axis=0)

        return y_pred
