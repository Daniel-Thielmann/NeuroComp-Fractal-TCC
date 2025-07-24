import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Importa filtro Chebyshev II do bciflow
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII


# Adiciona o diretório raiz ao path do Python para importações (padrão test_fractal.py)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from methods.features.logpower import logpower
from contexts.CBCIC import cbcic


def test_logpower_classification_wcci2020():
    """
    Classificação usando LogPower + MIBIF (8 features) para todos os sujeitos do WCCI2020.
    Calcula acurácia e kappa por sujeito usando LDA e StratifiedKFold 5-fold CV.
    Returns:
        dict: Resultados de performance por sujeito
    """

    def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=4):
        """ """

    print('Teste "LogPower" ("WCCI2020"):')
    import scipy.io as sio

    def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=4):
        eegdata = {"X": data, "sfreq": sfreq}
        eegdata_filtrado = chebyshevII(
            eegdata, low_cut=low_freq, high_cut=high_freq, order=order
        )
        return eegdata_filtrado["X"]

    results = {}
    all_accuracies = []
    all_kappas = []
    os.makedirs("results/wcci2020/logpower/evaluate", exist_ok=True)
    os.makedirs("results/wcci2020/logpower/training", exist_ok=True)
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
                print(
                    f"  AVISO: Poucos dados ({X.shape[0]} trials), pulando sujeito..."
                )
                continue
            X_filtered = bandpass_filter(X, 4, 40, sfreq=sfreq)
            eegdata = {"X": X_filtered}
            logpower_result = logpower(eegdata, flating=True)
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
            eval_df = pd.DataFrame(fold_results)
            eval_df.to_csv(
                f"results/wcci2020/logpower/evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            train_df = pd.DataFrame(fold_train_results)
            train_df.to_csv(
                f"results/wcci2020/logpower/training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"A{subject_id:02d}: acc={mean_accuracy:.4f}±{std_accuracy:.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print('Resumo "LogPower" ("WCCI2020"):')
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
    else:
        print("Nenhum resultado válido.")
    # Salva CSV geral
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
            "results/wcci2020/logpower/logpower_classification_results.csv", index=False
        )
        print("CSV salvo: results/wcci2020/logpower/")
    return results


if __name__ == "__main__":
    try:
        test_logpower_classification_wcci2020()
    except Exception as e:
        print(f"Erro: {str(e)}")
        sys.exit(1)
