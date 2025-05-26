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
from bciflow.modules.sf.csp import csp


def run_csp_logpower(subject_id: int):
    dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    # Filtra classes 1 e 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    eegdata = {"X": X[:, np.newaxis, :, :], "sfreq": 512}
    eegdata = filterbank(eegdata, kind_bp="chebyshevII")
    if not isinstance(eegdata, dict) or "X" not in eegdata:
        raise TypeError(
            f"Retorno inesperado de filterbank em csp_logpower: {type(eegdata)} - {eegdata}"
        )
    X_band = eegdata["X"]
    # Garante shape correto: [n_trials, n_bands, n_channels, n_samples]
    if X_band.ndim == 5:
        # Remove dimensao de filtros se existir
        X_band = X_band[:, :, 0, :, :]
    if X_band.shape[2] > X_band.shape[3]:
        # Se canais e samples estao invertidos, transpoe
        X_band = X_band.transpose(0, 1, 3, 2)

    # Aplica CSP
    transformer = csp()  # Não limita componentes no construtor
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})[
        "X"
    ]  # [trials, bands, components, samples]

    # Extrai features: log da potência média dos 2 primeiros componentes por banda
    features = []
    for trial in X_csp:
        # trial: [bands, components, samples]
        trial_feat = []
        for band in trial:
            # Seleciona apenas os 2 primeiros componentes
            comps = band[:2] if band.shape[0] >= 2 else band
            for comp in comps:
                trial_feat.append(np.log(np.mean(comp**2)))
        features.append(trial_feat)

    X_feat = np.array(features)
    X_feat = StandardScaler().fit_transform(X_feat)

    # Validacao cruzada para evitar overfitting
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
