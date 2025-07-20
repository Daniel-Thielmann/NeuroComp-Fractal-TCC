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

from methods.features.fractal import higuchi_fractal
from contexts.BCICIV2b import bciciv2b
from bciflow.modules.sf.csp import csp


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


def test_csp_fractal_classification_bciciv2b():
    print('Teste "CSP + Fractal" (BCICIV2b):')
    import scipy.io as sio
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, low_freq, high_freq, sfreq=250, order=4):
        nyquist = sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype="band")
        data_filtered = np.zeros_like(data)
        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                data_filtered[trial, channel, :] = filtfilt(
                    b, a, data[trial, channel, :]
                )
        return data_filtered

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
                results[f"P{subject_id:02d}"] = {"error": "Poucos trials"}
                continue
            X_filtered = bandpass_filter(X, 4, 40, int(sfreq))
            csp_transformer = csp()
            X_4d = X_filtered[:, np.newaxis, :, :]
            csp_transformer.fit({"X": X_4d, "y": y})
            X_csp_result = csp_transformer.transform({"X": X_4d})
            X_csp = X_csp_result["X"][:, 0, :, :]
            features = []
            for trial in X_csp:
                trial_feat = []
                for comp in trial:
                    comp_data = {"X": comp.reshape(1, 1, -1)}
                    fractal_result = higuchi_fractal(comp_data, flating=True, kmax=10)
                    fractal_dim = fractal_result["X"][0, 0]
                    energy = np.sum(comp**2)
                    std_val = np.std(comp)
                    trial_feat.extend([fractal_dim, energy, std_val])
                features.append(trial_feat)
            features = np.array(features)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf = LDA()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)
            mean_accuracy = np.mean(fold_accuracies)
            mean_kappa = np.mean(fold_kappas)
            results[f"P{subject_id:02d}"] = {
                "accuracy": mean_accuracy,
                "kappa": mean_kappa,
                "n_trials": X.shape[0],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            eval_df = pd.DataFrame(
                {
                    "Fold": list(range(1, 6)),
                    "Accuracy": fold_accuracies,
                    "Kappa": fold_kappas,
                }
            )
            os.makedirs("results/BCICIV2b/csp_fractal/evaluate", exist_ok=True)
            eval_df.to_csv(
                f"results/BCICIV2b/csp_fractal/evaluate/P{subject_id:02d}.csv",
                index=False,
            )
        except Exception as e:
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    for subject, metrics in results.items():
        if "error" in metrics:
            continue
        print(
            f"{subject}: acc={metrics['accuracy']:.4f}±{np.std(all_accuracies):.4f} | kappa={metrics['kappa']:.4f} | n_trials={metrics['n_trials']}"
        )
    print(f'Resumo "CSP + Fractal" (BCICIV2b):')
    print(
        f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
    )
    print("CSV salvo: results/BCICIV2b/csp_fractal/")
    return results


if __name__ == "__main__":
    test_csp_fractal_classification_bciciv2b()
