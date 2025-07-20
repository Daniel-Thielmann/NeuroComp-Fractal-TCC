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

from contexts.BCICIV2b import bciciv2b
from bciflow.modules.sf.csp import csp

import scipy.io


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    data_filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data_filtered[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])
    return data_filtered


def extract_fbcsp_logpower_features(X, y, frequency_bands, n_components=2):
    """Extrai features FBCSP + LogPower para BCICIV2b (3 canais)."""
    all_features = []

    for low_freq, high_freq in frequency_bands:
        # Filtra dados na banda de frequência
        X_filtered = bandpass_filter(X, low_freq, high_freq, 250, 5)

        # Aplica CSP
        csp_transformer = csp()
        X_4d = X_filtered[:, np.newaxis, :, :]  # [trials, bands, channels, samples]
        csp_transformer.fit({"X": X_4d, "y": y})
        X_csp_result = csp_transformer.transform({"X": X_4d})
        X_csp = X_csp_result["X"][:, 0, :, :]  # [trials, components, samples]

        # Seleciona até n_components mais discriminativos
        selected_comps = X_csp[:, : min(n_components, X_csp.shape[1]), :]

        # Calcula LogPower: log(sum(componente²) + 1e-10)
        band_features = []
        for trial in selected_comps:
            trial_feat = []
            for comp in trial:
                logpower = np.log(np.sum(comp**2) + 1e-10)
                trial_feat.append(logpower)
            band_features.append(trial_feat)

        all_features.append(np.array(band_features))

    # Concatena features de todas as bandas
    features = np.concatenate(all_features, axis=1)
    return features


def test_fbcsp_logpower_classification_bciciv2b():
    print('Teste "FBCSP + LogPower" ("BCICIV2b"):')

    # Define bandas de frequência
    frequency_bands = [
        (8, 12),  # Alpha baixo
        (12, 16),  # Alpha alto
        (16, 20),  # Beta baixo
        (20, 24),  # Beta médio
        (24, 30),  # Beta alto
    ]

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
            X = eegdata["X"]
            y = eegdata["y"]
            sfreq = eegdata["sfreq"]
            if X.ndim == 4 and X.shape[1] == 1:
                X = X.squeeze(1)
            if X.shape[0] < 10:
                continue
            features = extract_fbcsp_logpower_features(
                X, y, frequency_bands, n_components=2
            )
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            fold_results = []
            fold_train_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                mibif = SelectKBest(
                    score_func=mutual_info_classif, k=min(10, features.shape[1])
                )
                X_train_selected = mibif.fit_transform(X_train, y_train)
                X_test_selected = mibif.transform(X_test)
                fold_scaler = StandardScaler()
                X_train_final = fold_scaler.fit_transform(X_train_selected)
                X_test_final = fold_scaler.transform(X_test_selected)
                clf = LDA()
                clf.fit(X_train_final, y_train)
                y_pred = clf.predict(X_test_final)
                y_pred_train = clf.predict(X_train_final)
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)
                train_accuracy = (y_pred_train == y_train).mean()
                train_kappa = cohen_kappa_score(y_train, y_pred_train)
                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)
                fold_results.append(
                    {
                        "Fold": fold_idx + 1,
                        "Test_Accuracy": accuracy,
                        "Test_Kappa": kappa,
                    }
                )
                fold_train_results.append(
                    {
                        "Fold": fold_idx + 1,
                        "Train_Accuracy": train_accuracy,
                        "Train_Kappa": train_kappa,
                    }
                )
            mean_accuracy = np.mean(fold_accuracies)
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
            eval_df = pd.DataFrame(fold_results)
            os.makedirs("results/BCICIV2b/fbcsp_logpower/evaluate", exist_ok=True)
            os.makedirs("results/BCICIV2b/fbcsp_logpower/training", exist_ok=True)
            eval_df.to_csv(
                f"results/BCICIV2b/fbcsp_logpower/evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            train_df = pd.DataFrame(fold_train_results)
            train_df.to_csv(
                f"results/BCICIV2b/fbcsp_logpower/training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"P{subject_id:02d}: acc={mean_accuracy:.4f}±{np.std(fold_accuracies):.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception:
            results[f"P{subject_id:02d}"] = {"error": "erro"}
    print(f'Resumo "FBCSP + LogPower" ("BCICIV2b"):')
    print(
        f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
    )
    print("CSV salvo: results/BCICIV2b/fbcsp_logpower/")
    return results


if __name__ == "__main__":
    try:
        test_fbcsp_logpower_classification_bciciv2b()
    except Exception:
        sys.exit(1)
