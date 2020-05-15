# wczytanie zestawu danych
import numpy as np

dataset = 'ecoli-0-1-4-6_vs_5'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

# zdefiniowanie klasyfikatorów i technik preprocessingu
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour

clf = DecisionTreeClassifier(random_state=1410)
preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=1410),
    'smote' : SMOTE(random_state=1410),
    'rus': RandomUnderSampler(random_state=1410),
    'cnn': CondensedNearestNeighbour(random_state=1410),
}

from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

# walidacja krzyżowa
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)

# Zapisanie wynikow
np.save('results', scores)
