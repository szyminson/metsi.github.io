from sklearn import datasets

X, y = datasets.make_classification()

print(X.shape, y.shape)

X, y = datasets.make_classification(n_samples=1000, n_features=8)

print(X.shape, y.shape)

X, y = datasets.make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_repeated=0,
    n_redundant=0,
    flip_y=0.05,
    random_state=1410,
    n_clusters_per_class=1,
)

print(X.shape, y.shape)
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 2.5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.tight_layout()
plt.savefig("sroda1.png")
"""

import numpy as np

dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)

print(dataset.shape)

np.savetxt(
    "dataset.csv",
    dataset,
    delimiter=",",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"],
)

dataset = np.genfromtxt("dataset.csv", delimiter=",")

print(dataset)


X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

print(X, y)
