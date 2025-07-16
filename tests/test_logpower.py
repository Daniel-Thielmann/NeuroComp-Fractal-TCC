import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "contexts"))
)

from methods.features.logpower import logpower
from contexts.CBCIC import cbcic


def test_logpower_feature():
    """
    Testa a função logpower com dados simulados.

    Verifica se a função:
    - Executa sem erros
    - Produz output no formato correto
    - Não gera valores NaN ou infinitos
    """
    print("Testando função logpower...")

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
    Testa classificação usando features LogPower em todos os sujeitos do WCCI2020.

    Calcula acurácia e kappa para cada sujeito usando LDA com 5-fold cross-validation.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print("Testando classificação com LogPower no dataset WCCI2020...")
    print("Dataset: WCCI2020 (9 sujeitos)")
    print("Tarefa: Classificação de motor imagery usando features log power")
    print("Pipeline: Filtro (8-30Hz) → LogPower → Normalização → LDA (5-fold CV)")
    print("-" * 60)

    import scipy.io as sio
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=4):
        """Aplica filtro passa-banda usando Butterworth."""
        nyquist = sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, data, axis=-1)

    results = {}
    all_accuracies = []
    all_kappas = []

    for subject_id in range(1, 10):  # Sujeitos 1-9
        print(f"Processando sujeito P{subject_id:02d}...")

        try:
            # Carrega dados do WCCI2020
            mat_file = f"dataset/wcci2020/parsed_P{subject_id:02d}T.mat"
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {mat_file}")

            mat_data = sio.loadmat(mat_file)
            X = mat_data["RawEEGData"]  # [trials, channels, samples]
            y = mat_data["Labels"].flatten()  # [trials]
            sfreq = mat_data["sampRate"][0][0]  # Frequência de amostragem

            print(
                f"  Dados carregados: {X.shape[0]} trials, {X.shape[1]} canais, {X.shape[2]} amostras"
            )
            print(f"  Classes: {np.unique(y)}, Freq: {sfreq}Hz")

            if X.shape[0] < 10:
                print(
                    f"  AVISO: Poucos dados ({X.shape[0]} trials), pulando sujeito..."
                )
                continue

            # Aplica filtro passa-banda (8-30 Hz para motor imagery)
            X_filtered = bandpass_filter(X, 8, 30, sfreq=sfreq)

            # Extrai features log power
            eegdata = {"X": X_filtered}
            logpower_result = logpower(eegdata, flating=True)
            features = logpower_result["X"]  # [n_trials, n_channels]

            # Flatten features para matriz 2D
            features = features.reshape(features.shape[0], -1)

            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []

            for train_idx, test_idx in skf.split(features, y):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Normalização
                scaler = StandardScaler()
                X_train_final = scaler.fit_transform(X_train)
                X_test_final = scaler.transform(X_test)

                # Classificação com LDA
                clf = LDA()
                clf.fit(X_train_final, y_train)
                y_pred = clf.predict(X_test_final)

                # Calcula métricas do fold
                accuracy = (y_pred == y_test).mean()
                kappa = cohen_kappa_score(y_test, y_pred)

                fold_accuracies.append(accuracy)
                fold_kappas.append(kappa)

            # Média das métricas dos 5 folds
            mean_accuracy = np.mean(fold_accuracies)
            mean_kappa = np.mean(fold_kappas)

            # Armazena resultados
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
                f"  Acurácia: {mean_accuracy:.4f} | Kappa: {mean_kappa:.4f} | Trials: {X.shape[0]} | Features: {features.shape[1]}"
            )

        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}

    # Estatísticas gerais
    print("-" * 60)
    print("RESULTADOS FINAIS:")
    print(
        f"Acurácia média: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}"
    )
    print(f"Kappa médio: {np.mean(all_kappas):.4f} ± {np.std(all_kappas):.4f}")
    print(f"Melhor acurácia: {np.max(all_accuracies):.4f}")
    print(f"Pior acurácia: {np.min(all_accuracies):.4f}")

    print("Teste de classificação com LogPower concluído com sucesso.")

    return results


if __name__ == "__main__":
    print("=== TESTE: Função LogPower ===")

    # Teste básico com dados simulados
    print("\n1. Teste básico com dados simulados:")
    success = test_logpower_feature()
    if success:
        print("✓ Teste básico concluído com sucesso.")
    else:
        print("✗ Teste básico falhou.")
        sys.exit(1)

    # Teste com dataset WCCI2020
    print("\n2. Teste de classificação com dataset WCCI2020 (todos os sujeitos):")
    try:
        results = test_logpower_classification_wcci2020()

        # Salva resultados para análise posterior
        os.makedirs("results/test_outputs", exist_ok=True)

        # Converte resultados para DataFrame
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
                "results/test_outputs/logpower_wcci2020_classification_results.csv",
                index=False,
            )
            print(
                f"\n✓ Resultados salvos em: results/test_outputs/logpower_wcci2020_classification_results.csv"
            )

        print("✓ Teste completo de classificação com LogPower concluído com sucesso!")

    except Exception as e:
        print(f"✗ Erro no teste de classificação com dataset WCCI2020: {str(e)}")
        sys.exit(1)
