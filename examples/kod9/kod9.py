import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# Generowanie strumienia danych z uzyciem biblioteki stream learn
stream = sl.streams.StreamGenerator(n_chunks=200,
                                    chunk_size=500,
                                    n_classes=2,
                                    n_drifts=1,
                                    n_features=10,
                                    random_state=12345)

# Lista klasyfikatorow
clfs = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
    MLPClassifier(hidden_layer_sizes=(10)),
]
clf_names = [
    "SEA",
    "MLP",
]

# Wybrana metryka
metrics = [sl.metrics.f1_score,
           sl.metrics.geometric_mean_score_1]

metrics_names = ["F1 score",
                 "G-mean"]

# Inicjalizacja ewaluatora
evaluator = sl.evaluators.TestThenTrain(metrics)

# Uruchomienie
evaluator.process(stream, clfs)

# Rysowanie wykresu
fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.show()
