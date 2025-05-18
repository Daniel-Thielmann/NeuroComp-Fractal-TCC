import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import logpower


def run_fbcsp_logpower(subject_id: int):
    dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra classes 1 e 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    eegdata = {"X": X, "sfreq": 512}
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")
    if not isinstance(eegdata, dict) or "X" not in eegdata:
        raise TypeError(
            f"Retorno inesperado de filterbank em fbcsp_logpower: {type(eegdata)} - {eegdata}"
        )
    X_filt = eegdata["X"]

    if X_filt.ndim == 5:
        n_trials, n_bands, n_chans, n_filters, n_samples = X_filt.shape
        X_reshaped = X_filt.transpose(0, 1, 3, 2, 4).reshape(
            n_trials, n_bands * n_filters * n_chans, n_samples
        )
    elif X_filt.ndim == 4:
        n_trials, n_bands, n_chans, n_samples = X_filt.shape
        X_reshaped = X_filt.reshape(n_trials, n_bands * n_chans, n_samples)
    else:
        raise ValueError(f"Shape inesperado após filterbank: {X_filt.shape}")

    X_log = logpower(sfreq=512).extract(X_reshaped)

    features = []
    for trial in X_log:
        trial_feat = [np.log(np.mean(np.square(val))) for val in trial]
        features.append(trial_feat)

    X_feat = np.array(features)
    X_feat = StandardScaler().fit_transform(X_feat)

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
        clf = LDA()
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
