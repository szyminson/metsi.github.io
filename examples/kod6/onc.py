import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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

    def fit(self, X, y):
        # czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # przechowanie unikalnych klas problemu
        self.classes_ = np.unique(y)
        # zapamietujemy X i y
        self.X_, self.y_ = X, y
        # przygotowujemy narzedzie do liczenia dystansow
        self.dm_ = DistanceMetric.get_metric(self.metric)

        # kontener na centroidy klas
        self.centroids_ = []
        # plt.scatter(self.X_[:, 0], self.X_[:, 1], c=y, cmap='bwr')
        # plt.savefig("cos")
        # dla kazdej klasy
        for cl in self.classes_:
            # wybieramy tylko instancje nalezace do danej klasy
            X_class = self.X_[self.y_ == cl]

            # przynajmniej jeden obieg petli
            while True:
                # wyliczamy centroid klasy
                class_centroid = np.mean(X_class, axis=0)
                # liczymy odchylenie standardowe instancji klasy
                std = np.std(X_class, axis=0)

                # możliwie najdalej znajdująca się instancje
                self.borderline_ = class_centroid + (self.sigma * std)

                # maksymalny dopuszczalny dystans
                accepted_distances = np.squeeze(self.dm_.pairwise(
                    class_centroid.reshape(1, X_class.shape[1]), self.borderline_.reshape(1, X_class.shape[1])))

                # liczymy dystanse wszystkich obiektow klasy od centroidu
                distances = np.squeeze(self.dm_.pairwise(
                    class_centroid.reshape(1, X_class.shape[1]), X_class))

                # plt.scatter(class_centroid[0], class_centroid[1], c='black')
                # plt.savefig("cos")

                # uznajemy za outliery te instancje, ktore znajduja sie od
                # centroidu dalej niz 3 * std
                self.outliers_mask_ = np.array(distances > accepted_distances)
                # konczymy optymalizacje, jezeli nie mamy autlierow
                # lub nie chcielismy w ogole z niej korzystac
                if np.sum(self.outliers_mask_) == 0 or self.optimize == False:
                    break
                # w inym przypadku pozbywamy sie outlierow
                else:
                    # plt.scatter(X_class[self.outliers_mask_, 0], X_class[self.outliers_mask_, 1], c='grey')
                    # plt.savefig("cos")
                    X_class = X_class[self.outliers_mask_ == False]

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
        distance_pred = self.dm_.pairwise(self.centroids_, X)
        # uznajemy, ze instancje naleza do klasy, ktorej centroid znajduje
        # sie blizej
        y_pred = np.argmin(distance_pred, axis=0)
        # zwracamy predykcje
        return y_pred
