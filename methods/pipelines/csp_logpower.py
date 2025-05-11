import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import StratifiedKFold

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

    eegdata = {"X": X, "sfreq": 512}
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")
    X_band = eegdata["X"]  # [trials, bands, channels, samples]

    # Aplica CSP
    transformer = csp()
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})[
        "X"
    ]  # [trials, bands, components, samples]

    # Extrai features: log da potência média por componente
    features = []
    for trial in X_csp:
        trial_feat = [np.log(np.mean(comp**2)) for band in trial for comp in band]
        features.append(trial_feat)

    X_feat = np.array(features)
    X_feat = StandardScaler().fit_transform(X_feat)
    X_feat = PCA(n_components=min(15, X_feat.shape[1])).fit_transform(X_feat)

    # Validação cruzada para evitar overfitting
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
        clf = QDA(reg_param=0.1)
        clf.fit(X_feat[train_idx], y[train_idx])
        probs = clf.predict_proba(X_feat[test_idx])

        for i, idx in enumerate(test_idx):
            results.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
            )

    return results
