from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_repeated=0,
    n_redundant=0,
    flip_y=.05,
    random_state=1410,
    n_clusters_per_class=1
)

dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)

np.savetxt(
    "dataset.csv",
    dataset,
    delimiter=",",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"],
)

dataset = np.genfromtxt("dataset.csv", delimiter=",")

X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.3,
    random_state=42
)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

class_probabilities = clf.predict_proba(X_test)
print("Class probabilities:\n", class_probabilities)

predict = np.argmax(class_probabilities, axis=1)
print("Predicted labels:\n", predict)

print("True labels:     ", y_test)
print("Predicted labels:", predict)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predict)
print("Accuracy score:\n %.2f" % score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict)
print("Confusion matrix: \n", cm)
