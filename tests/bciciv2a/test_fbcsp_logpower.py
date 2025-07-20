import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.signal import butter, filtfilt


# Garante que o diretório raiz do projeto está no sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CONTEXTS = os.path.join(ROOT, "contexts")
if CONTEXTS not in sys.path:
    sys.path.append(CONTEXTS)

from contexts.BCICIV2a import bciciv2a
from bciflow.modules.sf.csp import csp

import scipy.io


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    from scipy.signal import cheby2

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby2(order, 20, [low, high], btype="band")
    data_filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data_filtered[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])
    return data_filtered


def extract_fbcsp_logpower_features(X, y, frequency_bands, n_components=4):
    """Extrai features FBCSP + LogPower."""
    all_features = []

    for low_freq, high_freq in frequency_bands:
        X_filtered = bandpass_filter(X, low_freq, high_freq, fs=250, order=4)
        csp_transformer = csp()
        X_4d = X_filtered[:, np.newaxis, :, :]
        csp_transformer.fit({"X": X_4d, "y": y})
        X_csp_result = csp_transformer.transform({"X": X_4d})
        X_csp = X_csp_result["X"][:, 0, :, :]
        selected_comps = (
            X_csp[:, :n_components, :] if X_csp.shape[1] >= n_components else X_csp
        )
        band_features = []
        for trial in selected_comps:
            trial_feat = []
            for comp in trial:
                logpower = np.log(np.mean(comp**2) + 1e-10)
                trial_feat.append(logpower)
            band_features.append(trial_feat)
        all_features.append(np.array(band_features))
    features = np.concatenate(all_features, axis=1)
    return features


def test_fbcsp_logpower_classification_bciciv2a():
    print('Teste "FBCSP + LogPower" ("BCICIV2a"):')
    frequency_banks = [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 28),
        (28, 32),
        (32, 36),
        (36, 40),
    ]
    results = {}
    all_accuracies = []
    all_kappas = []
    for subject_id in range(1, 10):
        try:
            eegdata = bciciv2a(
                subject=subject_id,
                session_list=["T"],
                labels=["left-hand", "right-hand"],
                path="dataset/BCICIV2a/",
            )
            X = eegdata["X"]
            y = eegdata["y"]
            sfreq = eegdata["sfreq"]
            if X.ndim == 4 and X.shape[1] == 1:
                X = X.squeeze(1)
            if X.shape[0] < 10:
                continue
            features = extract_fbcsp_logpower_features(
                X, y, frequency_banks, n_components=4
            )
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            for _, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                mibif = SelectKBest(
                    score_func=mutual_info_classif, k=min(8, features.shape[1])
                )
                X_train_selected = mibif.fit_transform(X_train, y_train)
                X_test_selected = mibif.transform(X_test)
                clf = LDA()
                clf.fit(X_train_selected, y_train)
                y_pred = clf.predict(X_test_selected)
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            mean_kappa = np.mean(fold_kappas)
            results[f"P{subject_id:02d}"] = {
                "accuracy": mean_accuracy,
                "kappa": mean_kappa,
                "n_trials": X.shape[0],
                "n_channels": X.shape[1],
                "n_samples": X.shape[2],
                "n_features": features.shape[1],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            print(
                f"P{subject_id:02d}: acc={mean_accuracy:.4f}{std_accuracy:.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print("Resumo FBCSP LogPower BCICIV2a:")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    print("Teste FBCSP + LogPower (BCICIV2a)")
    try:
        results = test_fbcsp_logpower_classification_bciciv2a()
        summary_data = []
        for subject, metrics in results.items():
            if "error" not in metrics:
                summary_data.append(
                    {
                        "Subject": subject,
                        "Accuracy": metrics["accuracy"],
                        "Kappa": metrics["kappa"],
                        "N_Trials": metrics["n_trials"],
                        "N_Channels": metrics["n_channels"],
                        "N_Samples": metrics["n_samples"],
                        "N_Features": metrics["n_features"],
                    }
                )
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                "results/BCICIV2a/fbcsp_logpower/fbcsp_logpower_classification_results.csv",
                index=False,
            )
            print(
                "CSV salvo: results/BCICIV2a/fbcsp_logpower/fbcsp_logpower_classification_results.csv"
            )
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)
