import numpy as np
import pandas as pd
import os
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from bciflow.modules.tf.filterbank import filterbank
from methods.features.higuchi import higuchi_fd
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def run_fbcsp_fractal(subject_id, data_path="dataset/wcci2020/"):
    # 1. Carrega os dados
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"]  # [n_trials, 1, channels, samples]
    y = np.array(dataset["y"])
    y = y + 1  # Ajusta para [1, 2]

    # 2. Aplica filtro bank (Chebyshev Type II)
    eeg_filtered, _ = filterbank({"X": X, "sfreq": 512}, kind_bp="chebyshevII")
    X_filtered = eeg_filtered["X"]

    # Shape após filtro: [n_trials, n_bandas, channels, samples]

    # 3. Aplica CSP por banda
    transformer = csp()
    transformer.fit({"X": X_filtered, "y": y})
    X_csp = transformer.transform({"X": X_filtered})["X"]
    # Shape: [n_trials, n_bandas, n_components, samples]

    # 4. Extrai Higuchi FD de cada componente CSP
    features = []
    for trial in X_csp:  # trial shape: [n_bandas, n_components, samples]
        trial_feat = []
        for band in trial:
            trial_feat.extend([higuchi_fd(component) for component in band])
        features.append(trial_feat)
    features = np.array(features)  # [n_trials, total_features]

    # 5. Classificação com LDA
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train, X_test = features[train_idx], features[test_idx]
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
