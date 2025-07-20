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


def test_logpower_feature():
    """
    Testa a função logpower com dados simulados.
    Verifica execução, formato e integridade dos dados.
    """
    print('Teste "LogPower" ("WCCI2020"):')

    # Dados de teste: 50 trials, 22 canais, 1000 amostras (simulando dados reais)
    X = np.random.randn(50, 22, 1000)
    eegdata = {"X": X}

    # Testa execução sem erro
    result = logpower(eegdata, flating=True)

    # Verificações de integridade
    assert "X" in result, "Resultado deve conter chave 'X'"
    assert result["X"].shape == (50, 22), f"Shape incorreto: {result['X'].shape}"
    assert not np.isnan(result["X"]).any(), "Resultado contém valores NaN"
    assert not np.isinf(result["X"]).any(), "Resultado contém valores infinitos"

    # Estatísticas descritivas
    mean_logpower = np.mean(result["X"])
    std_logpower = np.std(result["X"])
    min_logpower = np.min(result["X"])
    max_logpower = np.max(result["X"])

    print(f"Teste concluído com sucesso.")
    print(f"Estatísticas do log power:")
    print(f"  Média: {mean_logpower:.4f}")
    print(f"  Desvio padrão: {std_logpower:.4f}")
    print(f"  Mínimo: {min_logpower:.4f}")
    print(f"  Máximo: {max_logpower:.4f}")

    return True


def test_logpower_classification_wcci2020():
    """
    Classificação usando LogPower + MIBIF (8 features) para todos os sujeitos do WCCI2020.
    Calcula acurácia e kappa por sujeito usando LDA e StratifiedKFold 5-fold CV.
    Returns:
        dict: Resultados de performance por sujeito
    """
    print('Teste "LogPower" ("WCCI2020"):')

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
        # Print padrão por sujeito
        try:
            mat_file = f"dataset/wcci2020/parsed_P{subject_id:02d}T.mat"
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {mat_file}")
            mat_data = sio.loadmat(mat_file)
            X = mat_data["RawEEGData"]
            y = mat_data["Labels"].flatten()
            sfreq = mat_data["sampRate"][0][0] if "sampRate" in mat_data else 512
            # Mantém prints mínimos, apenas avisos relevantes
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

            # Seleção de características MIBIF (8 features)
            selector = SelectKBest(mutual_info_classif, k=8)
            features_selected = selector.fit_transform(features, y)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []
            fold_train_accuracies = []
            fold_train_kappas = []
            fold_results = []
            fold_train_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(features_selected, y)
            ):
                X_train, X_test = (
                    features_selected[train_idx],
                    features_selected[test_idx],
                )
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
                "n_features": features.shape[1],
            }
            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)
            # Salva CSV por sujeito para Evaluate (test)
            eval_df = pd.DataFrame(fold_results)
            os.makedirs("results/wcci2020/logpower/Evaluate", exist_ok=True)
            eval_df.to_csv(
                f"results/wcci2020/logpower/Evaluate/P{subject_id:02d}_evaluate.csv",
                index=False,
            )
            # Salva CSV por sujeito para Training (train)
            train_df = pd.DataFrame(fold_train_results)
            os.makedirs("results/wcci2020/logpower/Training", exist_ok=True)
            train_df.to_csv(
                f"results/wcci2020/logpower/Training/P{subject_id:02d}_training.csv",
                index=False,
            )
            print(
                f"A{subject_id:02d}: acc={mean_accuracy:.4f}±{np.std(fold_accuracies):.4f} | kappa={mean_kappa:.4f} | n_trials={X.shape[0]}"
            )
        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}
    print("Resumo LogPower (WCCI2020):")
    if all_accuracies:
        print(
            f"Acc média={np.mean(all_accuracies):.4f}±{np.std(all_accuracies):.4f} | Kappa média={np.mean(all_kappas):.4f}±{np.std(all_kappas):.4f}"
        )
        print(
            "CSV salvo: results/wcci2020/logpower/logpower_classification_results.csv"
        )
    else:
        print("Nenhum resultado válido.")
    return results


if __name__ == "__main__":
    print('Teste "LogPower" ("WCCI2020"):')
    success = test_logpower_feature()
    if success:
        print("✓ Teste básico concluído com sucesso.")
    else:
        print("✗ Teste básico falhou.")
        sys.exit(1)
    results = test_logpower_classification_wcci2020()
    os.makedirs("results/wcci2020/logpower", exist_ok=True)
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
