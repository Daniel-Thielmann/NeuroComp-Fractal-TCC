import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import logpower


def run_fbcsp_logpower(subject_id: int):
    # Carrega os dados
    dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra apenas classes 1 e 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    # Aplica filtro em bandas
    eegdata = {"X": X, "sfreq": 512}
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")

    # Extrai logpower por banda sem achatar
    X_log = logpower(eegdata, flating=False)["X"]  # [n_trials, bands, features]

    # Extração de features: log da potência média por banda e canal
    features = []
    for trial in X_log:
        trial_feat = [np.log(np.mean(band**2)) for band in trial]
        features.append(trial_feat)

    # Pré-processamento
    X_feat = np.array(features)
    X_feat = StandardScaler().fit_transform(X_feat)
    X_feat = PCA(n_components=min(15, X_feat.shape[1])).fit_transform(X_feat)

    # Classificador
    clf = QDA(reg_param=0.1)
    clf.fit(X_feat, y)
    probs = clf.predict_proba(X_feat)

    # Formata resultados
    results = [
        {
            "subject_id": subject_id,
            "true_label": y[i],
            "left_prob": probs[i][0],
            "right_prob": probs[i][1],
        }
        for i in range(len(y))
    ]

    return results
