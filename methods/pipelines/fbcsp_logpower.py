# filepath: d:\dev\EEG-TCC\methods\pipelines\fbcsp_logpower.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import LogPower


def run_fbcsp_logpower(subject_id: int, data_path="dataset/wcci2020/"):
    """
    Executa o metodo FBCSP (Filter Bank CSP) combinado com Log Power para classificacao de EEG.

    Args:
        subject_id: ID do sujeito a ser processado (1-9)
        data_path: Caminho para o diretorio com os dados

    Returns:
        Lista de dicionarios com os resultados de classificacao
    """
    # Carrega os dados
    dataset = cbcic(subject=subject_id, path=data_path)
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
            f"Retorno inesperado de filterbank: {type(eegdata)} - {eegdata}"
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
        raise ValueError(f"Shape inesperado apos filterbank: {X_filt.shape}")

    # Extrai features com LogPower
    X_log = LogPower(sfreq=512).extract(X_reshaped)

    # Normaliza as features
    X_feat = StandardScaler().fit_transform(X_log)

    # Validacao cruzada com 5 folds
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
