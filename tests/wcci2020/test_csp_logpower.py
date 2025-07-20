import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from sklearn.metrics import cohen_kappa_score
import scipy.io as sio


# Função de extração de features CSP + LogPower
def extract_csp_features(X, y, n_components=4):
    from bciflow.modules.sf.csp import csp

    transformer = csp()
    X_4d = X[:, np.newaxis, :, :]
    transformer.fit({"X": X_4d, "y": y})
    X_csp = transformer.transform({"X": X_4d})["X"]
    X_csp = X_csp[:, 0, :, :]  # [trials, components, samples]
    features = []
    for trial in X_csp:
        comps = trial[:n_components] if trial.shape[0] >= n_components else trial
        trial_features = []
        for component in comps:
            log_power = np.log(np.mean(component**2) + 1e-10)
            trial_features.append(log_power)
        features.append(trial_features)
    return np.array(features)


# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_csp_logpower_pipeline():
    """
    Testa o pipeline CSP com LogPower usando dataset WCCI2020 padronizado.

    Executa classificação de motor imagery para todos os sujeitos do dataset WCCI2020
    usando filtros, CSP e extração de features LogPower com validação cruzada robusta.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print('Teste "CSP + Logpower" ("WCCI2020"):')

    # imports já estão no topo do arquivo

    def bandpass_filter(data, low_freq=4, high_freq=40, sfreq=512, order=4):
        """Aplica filtro Chebyshev II do bciflow nos dados EEG."""
        eegdata = {"X": data, "sfreq": sfreq}
        eegdata_filtrado = chebyshevII(
            eegdata, low_cut=low_freq, high_cut=high_freq, order=order
        )
        return eegdata_filtrado["X"]

    results = {}
    all_accuracies = []
    all_kappas = []

    for subject_id in range(1, 10):  # Sujeitos 1-9
        print(f"Processando sujeito P{subject_id:02d}...")
        try:
            mat_file = f"dataset/wcci2020/parsed_P{subject_id:02d}T.mat"
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {mat_file}")
            mat_data = sio.loadmat(mat_file)
            X = mat_data["RawEEGData"]
            y = mat_data["Labels"].flatten()
            sfreq = mat_data["sampRate"][0][0] if "sampRate" in mat_data else 512
            # ...existing code...
            if X.shape[0] < 10:
                print(
                    f"  AVISO: Poucos dados ({X.shape[0]} trials), pulando sujeito..."
                )
                continue
            X_filtered = bandpass_filter(X, low_freq=4, high_freq=40, sfreq=sfreq)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            fold_train_accuracies = []
            fold_train_kappas = []
            fold_results = []
            fold_train_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_filtered, y)):
                X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                features_train = extract_csp_features(X_train, y_train, n_components=4)
                features_test = extract_csp_features(X_test, y_test, n_components=4)
                clf = LDA()
                clf.fit(features_train, y_train)
                y_pred = clf.predict(features_test)
                y_pred_train = clf.predict(features_train)
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)
                train_accuracy = (y_pred_train == y_train).mean()
                train_kappa = cohen_kappa_score(y_train, y_pred_train)
                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)
                fold_train_accuracies.append(train_accuracy)
                fold_train_kappas.append(train_kappa)
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
            mean_train_accuracy = np.mean(fold_train_accuracies)
            mean_train_kappa = np.mean(fold_train_kappas)
            results[f"P{subject_id:02d}"] = {
                "accuracy": mean_accuracy,
                "kappa": mean_kappa,
                "train_accuracy": mean_train_accuracy,
                "train_kappa": mean_train_kappa,
                "n_samples": X.shape[0],
                "cv_accuracies": fold_accuracies,
                "cv_kappas": fold_kappas,
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            # Salva CSV por sujeito para Evaluate (test)
            eval_df = pd.DataFrame(fold_results)
            os.makedirs("results/wcci2020/csp_logpower/Evaluate", exist_ok=True)
            eval_df.to_csv(
                f"results/wcci2020/csp_logpower/Evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            # Salva CSV por sujeito para Training (train)
            train_df = pd.DataFrame(fold_train_results)
            os.makedirs("results/wcci2020/csp_logpower/Training", exist_ok=True)
            train_df.to_csv(
                f"results/wcci2020/csp_logpower/Training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"A{subject_id:02d}: acc={mean_accuracy:.4f}±{np.std(fold_accuracies):.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print("Resumo CSP + Logpower (WCCI2020):")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
        print(
            "CSV salvo: results/wcci2020/csp_logpower/csp_logpower_classification_results.csv"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    print('Teste "CSP + Logpower" ("WCCI2020"):')
    results = test_csp_logpower_pipeline()
    os.makedirs("results/wcci2020/csp_logpower", exist_ok=True)
    summary_data = []
    for subject, metrics in results.items():
        if "error" not in metrics:
            summary_data.append(
                {
                    "Subject": subject,
                    "Accuracy": metrics["accuracy"],
                    "Kappa": metrics["kappa"],
                    "Train_Accuracy": metrics.get("train_accuracy", None),
                    "Train_Kappa": metrics.get("train_kappa", None),
                    "N_Samples": metrics["n_samples"],
                }
            )
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            "results/wcci2020/csp_logpower/csp_logpower_classification_results.csv",
            index=False,
        )
