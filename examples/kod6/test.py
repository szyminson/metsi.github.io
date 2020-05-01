import numpy as np
from onc import OptimizedNearestCentroid
from ooo import OOO
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# dataset = 'australian'
# dataset = np.genfromtxt("../%s.csv" % (dataset), delimiter=",")
# X = dataset[:, :-1]
# y = dataset[:, -1].astype(int)

X, y = make_classification(
    n_samples=700,
    n_features=2,
    n_informative=2,
    n_repeated=0,
    n_redundant=0,
    flip_y=.15,
    random_state=32456,
    n_clusters_per_class=1,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.3,
    random_state=42
)

clf = OptimizedNearestCentroid(metric='euclidean', optimize=False, sigma=3)
clfo = OptimizedNearestCentroid(metric='euclidean', optimize=True, sigma=3)
ooo = OOO(metric="euclidean", sigma=3)
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
start1 = time.time()
clfo.fit(X_train, y_train)
end1 = time.time()
start2 = time.time()
ooo.fit(X_train, y_train)
end2 = time.time()

print("czas : ", end - start)
print("czas1: ", end1 - start1)
print("czas2: ", end2 - start2)

pred = clf.predict(X_test)
predo = clfo.predict(X_test)
predoo = ooo.predict(X_test)

print("Zwykly:         ", accuracy_score(y_test, pred))
print("Optymalizowany: ", accuracy_score(y_test, predo))
print("OOO           : ", accuracy_score(y_test, predoo))
