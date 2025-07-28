import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CONTEXTS = os.path.join(ROOT, "contexts")
if CONTEXTS not in sys.path:
    sys.path.append(CONTEXTS)
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from contexts.BCICIV2b import bciciv2b
from methods.features.logpower import logpower
from scipy.signal import cheby2, filtfilt

# Garante que o diretório raiz do projeto está no sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CONTEXTS = os.path.join(ROOT, "contexts")
if CONTEXTS not in sys.path:
    sys.path.append(CONTEXTS)


def chebyshev2_bandpass(data, lowcut, highcut, fs, order=4, rs=20):
    fs = int(fs)
    order = int(order)
    rs = int(rs)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby2(order, rs, [low, high], btype="bandpass")
    # Suporta shape [n_trials, 1, n_channels, n_samples]
    data_filtered = np.zeros_like(data)
    if data.ndim == 4:
        for trial in range(data.shape[0]):
            for block in range(data.shape[1]):
                for channel in range(data.shape[2]):
                    data_filtered[trial, block, channel, :] = filtfilt(
                        b, a, data[trial, block, channel, :]
                    )
    elif data.ndim == 3:
        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                data_filtered[trial, channel, :] = filtfilt(
                    b, a, data[trial, channel, :]
                )
    else:
        raise ValueError("Formato de dados não suportado para filtragem.")
    return data_filtered


def test_logpower_classification_bciciv2b():
    print('Teste "LogPower" ("BCICIV2b"):')

    results = {}
    all_accuracies = []
    all_kappas = []
    for subject_id in range(1, 10):
        try:
            eegdata = bciciv2b(
                subject=subject_id,
                session_list=["01T", "02T", "03T", "04E", "05E"],
                labels=["left-hand", "right-hand"],
                path="dataset/BCICIV2b/",
            )
            X = eegdata.X
            y = eegdata.y
            sfreq = eegdata.sfreq
            # Squeeze para shape [n_trials, n_channels, n_samples] igual ao bciflow
            if X.ndim == 4 and X.shape[1] == 1:
                X = np.squeeze(X, axis=1)
            if X.shape[0] < 10:
                continue
            X_filtered = chebyshev2_bandpass(X, lowcut=4, highcut=40, fs=sfreq)
            if X_filtered.ndim == 4 and X_filtered.shape[1] == 1:
                X_filtered = np.squeeze(X_filtered, axis=1)
            logpower_result = logpower({"X": X_filtered}, flating=True)
            features = logpower_result["X"]
            features = features.reshape(features.shape[0], -1)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            fold_results = []
            fold_train_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf = LDA()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)
                train_accuracy = (y_pred_train == y_train).mean()
                train_kappa = cohen_kappa_score(y_train, y_pred_train)
                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)
                fold_results.append(
                    {
                        "Fold": fold_idx + 1,
                        "Accuracy": accuracy,
                        "Kappa": kappa,
                        "N_Samples": len(y_test),
                    }
                )
                fold_train_results.append(
                    {
                        "Fold": fold_idx + 1,
                        "Accuracy": train_accuracy,
                        "Kappa": train_kappa,
                        "N_Samples": len(y_train),
                    }
                )
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
                f"P{subject_id:02d}: acc={mean_accuracy:.4f}±{std_accuracy:.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
            # Salva CSVs por sujeito no padrão WCCI2020
            os.makedirs("results/BCICIV2b/logpower/evaluate", exist_ok=True)
            os.makedirs("results/BCICIV2b/logpower/training", exist_ok=True)
            eval_df = pd.DataFrame(fold_results)
            eval_df.to_csv(
                f"results/BCICIV2b/logpower/evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            train_df = pd.DataFrame(fold_train_results)
            train_df.to_csv(
                f"results/BCICIV2b/logpower/training/P{subject_id:02d}_training.csv",
                index=False,
            )
        except Exception as e:
            results[f"P{subject_id:02d}"] = {"error": str(e)}

    print('Resumo "LogPower" ("BCICIV2b"):')
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
    else:
        print("Nenhum resultado válido.")
    # Salva CSV geral
    if results:
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
                "results/BCICIV2b/logpower/logpower_classification_results.csv",
                index=False,
            )
            print(
                "CSV salvo: results/BCICIV2b/logpower/logpower_classification_results.csv"
            )
    return results


if __name__ == "__main__":
    try:
        test_logpower_classification_bciciv2b()
    except Exception:
        sys.exit(1)
