"""
Features adicionais utilizadas:
energia = np.sum(comp ** 2) calcula a energia do sinal do componente comp, ou seja, a soma dos quadrados dos valores desse vetor (representa a "força" ou "potência" do sinal).
std = np.std(comp) calcula o desvio padrão do componente comp, ou seja, a dispersão dos valores em torno da média (representa a variabilidade do sinal).
Essas duas métricas são usadas como features adicionais para cada componente extraído do EEG, aumentando o poder discriminativo dos classificadores nos pipelines de classificação.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "contexts"))
)

from methods.features.fractal import higuchi_fractal
from contexts.CBCIC import cbcic
from bciflow.modules.sf.csp import csp


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Aplica filtro Chebyshev II do bciflow nos dados EEG.
    Args:
        data: array [trials, channels, samples]
        lowcut: frequência de corte inferior (Hz)
        highcut: frequência de corte superior (Hz)
        fs: frequência de amostragem (Hz)
        order: ordem do filtro
    Returns:
        data_filtrada: dados filtrados
    """
    eegdata = {"X": data, "sfreq": fs}
    eegdata_filtrado = chebyshevII(
        eegdata, low_cut=lowcut, high_cut=highcut, order=order
    )
    return eegdata_filtrado["X"]


def test_csp_fractal_classification_wcci2020():
    """
    Testa classificação usando CSP + Higuchi Fractal em todos os sujeitos do WCCI2020.

    Calcula acurácia e kappa para cada sujeito usando CSP + Fractal + LDA com 5-fold cross-validation.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print("Teste CSP + Fractal (WCCI2020):")

    import scipy.io as sio

    def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=4):
        """
        Aplica filtro Chebyshev II do bciflow nos dados EEG.
        """
        eegdata = {"X": data, "sfreq": sfreq}
        eegdata_filtrado = chebyshevII(
            eegdata, low_cut=low_freq, high_cut=high_freq, order=order
        )
        return eegdata_filtrado["X"]

    results = {}
    all_accuracies = []
    all_kappas = []

    for subject_id in range(1, 10):  # Sujeitos 1-9
        try:
            mat_file = f"dataset/wcci2020/parsed_P{subject_id:02d}T.mat"
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {mat_file}")
            mat_data = sio.loadmat(mat_file)
            X = mat_data["RawEEGData"]
            y = mat_data["Labels"].flatten()
            sfreq = mat_data["sampRate"][0][0] if "sampRate" in mat_data else 512
            if X.shape[0] < 10:
                print(f"AVISO: Poucos dados ({X.shape[0]} trials), pulando sujeito...")
                continue
            X_filtered = bandpass_filter(X, 4, 40, sfreq=sfreq)
            X_4d = X_filtered[:, np.newaxis, :, :]
            csp_transformer = csp()
            csp_transformer.fit({"X": X_4d, "y": y})
            X_csp_result = csp_transformer.transform({"X": X_4d})
            X_csp = X_csp_result["X"][:, 0, :, :]
            features = []
            for trial in X_csp:
                comps = trial[:4] if trial.shape[0] >= 4 else trial
                trial_feat = []
                for comp in comps:
                    comp_data = {"X": comp.reshape(1, 1, -1)}
                    fractal_result = higuchi_fractal(comp_data, flating=True)
                    fractal_dim = fractal_result["X"][0, 0]
                    # energia = np.sum(comp ** 2) calcula a energia do sinal do componente comp, ou seja, a soma dos quadrados dos valores desse vetor (representa a "força" ou "potência" do sinal).
                    energia = np.sum(comp**2)
                    # std = np.std(comp) calcula o desvio padrão do componente comp, ou seja, a dispersão dos valores em torno da média (representa a variabilidade do sinal).
                    std = np.std(comp)
                    trial_feat.extend([fractal_dim, energia, std])
                features.append(trial_feat)
            features = np.array(features)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            fold_train_accuracies = []
            fold_train_kappas = []
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
                "n_trials": X.shape[0],
                "n_channels": X.shape[1],
                "n_samples": X.shape[2],
                "n_csp_components": X_csp.shape[1],
                "n_features": features.shape[1],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            # Salva CSV por sujeito para Evaluate (test)
            eval_df = pd.DataFrame(fold_results)
            os.makedirs("results/wcci2020/csp_fractal/Evaluate", exist_ok=True)
            eval_df.to_csv(
                f"results/wcci2020/csp_fractal/Evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            # Salva CSV por sujeito para Training (train)
            train_df = pd.DataFrame(fold_train_results)
            os.makedirs("results/wcci2020/csp_fractal/Training", exist_ok=True)
            train_df.to_csv(
                f"results/wcci2020/csp_fractal/Training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"A{subject_id:02d}: acc={mean_accuracy:.4f}±{np.std(fold_accuracies):.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print("Resumo CSP + Fractal (WCCI2020):")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
        print(
            "CSV salvo: results/wcci2020/csp_fractal/csp_fractal_classification_results.csv"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    print('Teste "CSP + Fractal" ("WCCI2020"):')
    results = test_csp_fractal_classification_wcci2020()
    os.makedirs("results/wcci2020/csp_fractal", exist_ok=True)
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
                    "N_CSP_Components": metrics["n_csp_components"],
                    "N_Features": metrics["n_features"],
                }
            )
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            "results/wcci2020/csp_fractal/csp_fractal_classification_results.csv",
            index=False,
        )
