import numpy as np
import pandas as pd
import os
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import logpower
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def run_fbcsp_logpower(subject_id, data_path="dataset/wcci2020/"):
    # 1. Carrega os dados
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"]  # [n_trials, 1, channels, samples]
    y = np.array(dataset["y"]) + 1  # Ajusta rótulos para [1, 2]

    # 2. Aplica o Filter Bank
    X_filtered = filterbank({"X": X, "sfreq": 512}, kind_bp="chebyshevII")[0]["X"]

    # [n_trials, bands, ch, samples]

    # 3. Aplica CSP por banda
    transformer = csp()
    transformer.fit({"X": X_filtered, "y": y})
    X_csp = transformer.transform({"X": X_filtered})[
        "X"
    ]  # [n_trials, bands, components, samples]

    # 4. Prepara formato para logpower: reshape para [n_trials, 1, all_components, samples]
    n_trials, n_bands, n_comp, n_samples = X_csp.shape
    X_reshaped = X_csp.transpose(0, 2, 1, 3).reshape(n_trials, 1, -1, n_samples)

    # 5. Aplica logpower
    X_log = logpower({"X": X_reshaped}, flating=True)["X"]  # [n_trials, features]

    # 6. Classificação
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_log, y)):
        X_train, X_test = X_log[train_idx], X_log[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LDA()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)

        for i, idx in enumerate(test_idx):
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y_test[i],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
            )

    return rows
