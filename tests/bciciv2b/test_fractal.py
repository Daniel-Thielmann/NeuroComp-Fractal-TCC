import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt


# Garante que o diretório raiz do projeto está no sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CONTEXTS = os.path.join(ROOT, "contexts")
if CONTEXTS not in sys.path:
    sys.path.append(CONTEXTS)

from contexts.BCICIV2b import bciciv2b
from methods.features.fractal import higuchi_fractal

import scipy.io


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import cheby2

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby2(order, 20, [low, high], btype="bandpass")
    data_filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data_filtered[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])
    return data_filtered


def test_fractal_classification_bciciv2b():
    print('Teste "Fractal" ("BCICIV2b"):')
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
            if X.shape[0] < 10:
                continue
            X_filtered = bandpass_filter(X.squeeze(), 4, 40, int(sfreq))
            eegdata = {"X": X_filtered}
            features_result = higuchi_fractal(eegdata, flating=True, kmax=10)
            fractal_features = features_result["X"]
            features = fractal_features.reshape(fractal_features.shape[0], -1)
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
            mean_kappa = np.mean(fold_kappas)
            results[f"P{subject_id:02d}"] = {
                "accuracy": mean_accuracy,
                "kappa": mean_kappa,
                "n_trials": X.shape[0],
                "n_channels": X.shape[1] if X.ndim > 2 else 1,
                "n_samples": X.shape[-1],
                "n_features": features.shape[1],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            eval_df = pd.DataFrame(fold_results)
            train_df = pd.DataFrame(fold_train_results)
            os.makedirs("results/BCICIV2b/fractal/evaluate", exist_ok=True)
            os.makedirs("results/BCICIV2b/fractal/training", exist_ok=True)
            eval_df.to_csv(
                f"results/BCICIV2b/fractal/evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            train_df.to_csv(
                f"results/BCICIV2b/fractal/training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"P{subject_id:02d}: acc={mean_accuracy:.4f}±{np.std(fold_accuracies):.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception:
            results[f"P{subject_id:02d}"] = {"error": "erro"}
    # Resumo final
    print(f'Resumo "Fractal" ("BCICIV2b"):')
    print(
        f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
    )
    print("CSV salvo: results/BCICIV2b/fractal/")
    # CSV geral
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
            "results/BCICIV2b/fractal/fractal_classification_results.csv", index=False
        )
    return results


if __name__ == "__main__":
    print("=== TESTE: Higuchi Fractal Dimension ===")

    print("\n2. Teste de classificação com dataset BCICIV2b (todos os sujeitos):")
    try:
        test_fractal_classification_bciciv2b()
        print(
            "Teste completo de classificação com Higuchi Fractal concluído com sucesso!"
        )
    except Exception as e:
        print(f"Erro no teste de classificação com dataset BCICIV2b: {str(e)}")
        sys.exit(1)
