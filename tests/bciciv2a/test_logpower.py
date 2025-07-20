import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt


# Garante que o diretório raiz do projeto está no sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CONTEXTS = os.path.join(ROOT, "contexts")
if CONTEXTS not in sys.path:
    sys.path.append(CONTEXTS)

from methods.features.logpower import logpower
from contexts.BCICIV2a import bciciv2a

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


def test_logpower_classification_bciciv2a():
    print('Teste "LogPower" ("BCICIV2a"):')
    results = {}
    all_accuracies = []
    all_kappas = []
    for subject_id in range(1, 10):
        try:
            eegdata = bciciv2a(
                subject=subject_id,
                session_list=["T"],
                labels=["left-hand", "right-hand", "both-feet", "tongue"],
                path="dataset/BCICIV2a/",
            )
            X = eegdata["X"]
            y = eegdata["y"]
            sfreq = eegdata["sfreq"]
            if X.ndim == 4 and X.shape[1] == 1:
                X = X.squeeze(1)
            if X.shape[0] < 10:
                continue
            X_filtered = bandpass_filter(X, 4, 40, fs=sfreq)
            eegdata = {"X": X_filtered}
            logpower_result = logpower(eegdata, flating=True)
            features = logpower_result["X"]
            features = features.reshape(features.shape[0], -1)
            mibif = SelectKBest(
                score_func=mutual_info_classif, k=min(8, features.shape[1])
            )
            features_selected = mibif.fit_transform(features, y)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            for _, (train_idx, test_idx) in enumerate(skf.split(features_selected, y)):
                X_train, X_test = (
                    features_selected[train_idx],
                    features_selected[test_idx],
                )
                y_train, y_test = y[train_idx], y[test_idx]
                clf = LDA()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
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
                "n_features": features_selected.shape[1],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            print(
                f"P{subject_id:02d}: acc={mean_accuracy:.4f}{std_accuracy:.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print("Resumo LogPower BCICIV2a:")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    try:
        results = test_logpower_classification_bciciv2a()
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
                "results/BCICIV2a/logpower/logpower_classification_results.csv",
                index=False,
            )
            print(
                "CSV salvo: results/BCICIV2a/logpower/logpower_classification_results.csv"
            )
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)
