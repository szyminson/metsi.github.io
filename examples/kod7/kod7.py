import numpy as np
from rse import RandomSubspaceEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

dataset = 'ionosphere'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

print("Total number of features", X.shape[1])

n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

# Eksepryment sprawdzajacy wplyw trybu glosowania

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Soft voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

# Eksepryment sprawdzajacy wplyw liczby cech

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=10, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("10/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("15/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=20, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("20/34 features - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

# Eksepryment sprawdzajacy wplyw liczby estymatorow

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=5, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("5 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("10 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=15, n_subspace_features=15, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("15 estimators - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
