# filepath: d:\dev\EEG-TCC\tests\test_logpower.py
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bciflow.datasets import cbcic
from bciflow.modules.tf.filterbank import filterbank
from methods.features.logpower import LogPower


def run_logpower_all():
    all_rows = []

    for subject_id in tqdm(range(1, 10), desc="Logpower"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"]  # [n_trials, 1, channels, samples]
        y = np.array(dataset["y"]) + 1

        # Filtra labels 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        eegdata_dict = {"X": X[:, np.newaxis, :, :], "sfreq": 512}
        eegdata_dict = filterbank(eegdata_dict, kind_bp="chebyshevII")
        if not isinstance(eegdata_dict, dict) or "X" not in eegdata_dict:
            raise TypeError(
                f"Retorno inesperado de filterbank: {type(eegdata_dict)} - {eegdata_dict}"
            )
        X_filtered = eegdata_dict["X"]

        if X_filtered.ndim != 5:
            raise ValueError(f"Shape inesperado ap�s filterbank: {X_filtered.shape}")

        n_trials, n_bands, n_chans, n_filters, n_samples = X_filtered.shape
        X_reshaped = X_filtered.transpose(0, 1, 3, 2, 4).reshape(
            n_trials, n_bands * n_filters * n_chans, n_samples
        )

        extractor = LogPower(sfreq=512)
        X_feat = extractor.extract(X_reshaped)
        X_feat = StandardScaler().fit_transform(X_feat)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
            clf = LDA()
            clf.fit(X_feat[train_idx], y[train_idx])
            probs = clf.predict_proba(X_feat[test_idx])

            for i, idx in enumerate(test_idx):
                row = {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                    "predicted": np.argmax(probs[i]) + 1,
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def test_logpower_accuracy():
    """Testa se a acur�cia do m�todo LogPower est� acima de um limiar aceit�vel."""
    df = run_logpower_all()
    acc = (df["true_label"] == df["predicted"]).mean()
    print(f"LogPower Accuracy: {acc:.4f}")
    assert acc > 0.6, f"Accuracy abaixo do esperado: {acc:.4f}"


def test_logpower_class_balance():
    """Testa se o balanceamento de classes est� adequado."""
    df = run_logpower_all()
    class_counts = df["true_label"].value_counts()
    print(f"Class distribution: {class_counts}")
    assert (
        class_counts.min() / class_counts.max() > 0.8
    ), f"Desbalanceamento excessivo: {class_counts}"


if __name__ == "__main__":
    df = run_logpower_all()
    acc = (df["true_label"] == df["predicted"]).mean()
    print(f"LogPower Accuracy: {acc:.4f}")
    print(f"Class distribution: {df['true_label'].value_counts()}")
