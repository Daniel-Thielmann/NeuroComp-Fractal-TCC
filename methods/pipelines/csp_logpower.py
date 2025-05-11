import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from bciflow.modules.sf.csp import csp


def run_csp_logpower(subject_id: int):
    dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra classes 1 e 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    # Filtro em bandas — mantém formato original [trials, bands, channels, samples]
    eegdata = {"X": X, "sfreq": 512}
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")
    X_band = eegdata["X"]

    # Aplica CSP
    transformer = csp()
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})[
        "X"
    ]  # [n_trials, bands, components, samples]

    # Extrai features: log da potência média por componente
    features = []
    for trial in X_csp:
        trial_feat = [np.log(np.mean(comp**2)) for band in trial for comp in band]
        features.append(trial_feat)

    X_feat = np.array(features)
    X_feat = StandardScaler().fit_transform(X_feat)
    X_feat = PCA(n_components=min(15, X_feat.shape[1])).fit_transform(X_feat)

    clf = QDA(reg_param=0.1)
    clf.fit(X_feat, y)
    probs = clf.predict_proba(X_feat)

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
