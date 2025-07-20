import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import cohen_kappa_score


def test_fbcsp_logpower_pipeline():
    """
    Testa o pipeline FBCSP com LogPower usando dataset WCCI2020 padronizado.

    Executa classificação de motor imagery para todos os sujeitos do dataset WCCI2020
    usando Filter Bank, CSP, extração de features LogPower e MIBIF com validação cruzada robusta.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print('Teste "FBCSP + Logpower" ("WCCI2020"):')

    import scipy.io as sio
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from scipy.signal import butter, filtfilt

    def chebyshev_bandpass_filter(data, lowcut, highcut, fs=512, order=4):
        from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII

        eegdata = {"X": data, "sfreq": fs}
        eegdata_filtrado = chebyshevII(
            eegdata, low_cut=lowcut, high_cut=highcut, order=order
        )
        return eegdata_filtrado["X"]

    def extract_fbcsp_logpower_features(X, y, filter_banks, n_components=4):
        from bciflow.modules.sf.csp import csp

        all_features = []
        for lowcut, highcut in filter_banks:
            X_band = chebyshev_bandpass_filter(X, lowcut, highcut, fs=512, order=4)
            csp_transformer = csp()
            csp_transformer.fit({"X": X_band[:, np.newaxis, :, :], "y": y})
            X_csp = csp_transformer.transform({"X": X_band[:, np.newaxis, :, :]})["X"]
            X_csp = X_csp[:, 0, :, :]  # [trials, components, samples]
            for trial_idx in range(X_csp.shape[0]):
                if trial_idx >= len(all_features):
                    all_features.append([])
                trial_components = X_csp[trial_idx]
                comps = (
                    trial_components[:n_components]
                    if trial_components.shape[0] >= n_components
                    else trial_components
                )
                for component in comps:
                    log_power = np.log(np.mean(component**2) + 1e-10)
                    all_features[trial_idx].append(log_power)
        return np.array(all_features)

    results = {}
    all_accuracies = []
    all_kappas = []

    # Define bandas de frequência (mesmo padrão do FBCSP+Fractal)
    filter_banks = [
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

    for subject_id in range(1, 10):
        try:
            mat_file = f"dataset/wcci2020/parsed_P{subject_id:02d}T.mat"
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {mat_file}")
            mat_data = sio.loadmat(mat_file)
            X = mat_data["RawEEGData"]
            y = mat_data["Labels"].flatten()
            sfreq = mat_data["sampRate"][0][0] if "sampRate" in mat_data else 512
            if X.shape[0] < 10:
                continue
            cv_accuracies = []
            cv_kappas = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                features_train = extract_fbcsp_logpower_features(
                    X_train, y_train, filter_banks, n_components=4
                )
                features_test = extract_fbcsp_logpower_features(
                    X_test, y_test, filter_banks, n_components=4
                )
                n_selected_features = min(8, features_train.shape[1])
                selector = SelectKBest(
                    score_func=mutual_info_classif, k=n_selected_features
                )
                features_train_final = selector.fit_transform(features_train, y_train)
                features_test_final = selector.transform(features_test)
                clf = LDA()
                clf.fit(features_train_final, y_train)
                y_pred = clf.predict(features_test_final)
                fold_accuracy = (y_test == y_pred).mean()
                fold_kappa = cohen_kappa_score(y_test, y_pred)
                cv_accuracies.append(fold_accuracy)
                cv_kappas.append(fold_kappa)
            accuracy = np.mean(cv_accuracies)
            kappa = np.mean(cv_kappas)
            results[f"P{subject_id:02d}"] = {
                "accuracy": accuracy,
                "kappa": kappa,
                "n_trials": X.shape[0],
                "n_channels": X.shape[1],
                "n_samples": X.shape[2],
                "cv_accuracies": cv_accuracies,
                "cv_kappas": cv_kappas,
            }
            all_accuracies.append(accuracy)
            all_kappas.append(kappa)
            os.makedirs("results/wcci2020/fbcsp_logpower/Evaluate", exist_ok=True)
            eval_df = pd.DataFrame(
                {
                    "Fold": list(range(1, 6)),
                    "Test_Accuracy": cv_accuracies,
                    "Test_Kappa": cv_kappas,
                }
            )
            eval_df.to_csv(
                f"results/wcci2020/fbcsp_logpower/Evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            print(
                f"A{subject_id:02d}: acc={accuracy:.4f}±{np.std(cv_accuracies):.4f} | kappa={kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            print(f"A{subject_id:02d}: ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}

    # Estatísticas gerais
    print("Resumo FBCSP + Logpower (WCCI2020):")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
        print(
            "CSV salvo: results/wcci2020/fbcsp_logpower/fbcsp_logpower_classification_results.csv"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    results = test_fbcsp_logpower_pipeline()
    os.makedirs("results/wcci2020/fbcsp_logpower", exist_ok=True)
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
                }
            )
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            "results/wcci2020/fbcsp_logpower/fbcsp_logpower_classification_results.csv",
            index=False,
        )
        print(
            "CSV salvo: results/wcci2020/fbcsp_logpower/fbcsp_logpower_classification_results.csv"
        )
