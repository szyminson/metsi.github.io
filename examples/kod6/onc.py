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
        # zapamietujemy X i y
        self.X_, self.y_ = X, y

        # kontener na centroidy klas
        self.centroids_ = []
        plt.scatter(self.X_[:, 0], self.X_[:, 1], c=y, cmap='bwr')
        # dla kazdej klasy
        for cl in self.classes_:
            # wybieramy tylko instancje nalezace do danej klasy
            X_class = self.X_[self.y_ == cl]
            # przynajmniej jeden obied petli
            self.optimize_ = True
            while self.optimize_:
                # jezeli mamy outliery to je usuwamy, od 2 obiegu petli
                if hasattr(self, 'outliers_') and self.optimize == True:
                    plt.scatter(X_class[self.outliers_, 0], X_class[self.outliers_, 1], c='gray')

                    X_class = np.delete(X_class, self.outliers_, axis=0)

                # wyliczamy centroid klasy
                class_centroid = np.mean(X_class, axis=0)
                # liczymy dystanse wszystkich obiektow klasy od centroidu
                distances = np.squeeze(self.dm.pairwise(
                    class_centroid.reshape(1, X_class.shape[1]), X_class))
                # liczymy odchylenie standardowe instancji klasy
                std = np.std(X_class)

                plt.scatter(class_centroid[0], class_centroid[1], c='black')

                # uznajemy za outliery te instancje, ktore znajduja sie od
                # centroidu dalej niz 3 * std
                self.outliers_ = np.squeeze(np.argwhere(distances > self.sigma * std))

                plt.savefig("cos")

                # konczymy optymalizacje, jezeli nie mamy autlierow
                # lub nie chcielismy w ogole z niej korzystac
                if self.outliers_.size == 0 or self.optimize == False:
                    self.optimize_ = False
            # dodajemy wyliczony centroid do listy
            self.centroids_.append(class_centroid)
        # zwracamy klasyfikator
        return self

    def predict(self, X):
        # sprawdzenie do wywolany zostal fit
        check_is_fitted(self)
        # sprawdzenie wejscia
        X = check_array(X)

        # liczymy dystanse instancji testowych od centroidow
        distance_pred= self.dm.pairwise(self.centroids_, X)
        # uznajemy, ze instancje naleza do klsy, ktorej centroid znajduje
        # sie blizej
        y_pred = np.argmin(distance_pred, axis=0)
        # zwracamy predykcje
        return y_pred
