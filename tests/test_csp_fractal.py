import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

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
    Aplica filtro passa-banda Butterworth nos dados EEG.

    Args:
        data: array [trials, channels, samples]
        lowcut: frequência de corte inferior (Hz)
        highcut: frequência de corte superior (Hz)
        fs: frequência de amostragem (Hz)
        order: ordem do filtro

    Returns:
        data_filtered: dados filtrados
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    # Aplica filtro em cada trial e canal
    data_filtered = np.zeros_like(data)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data_filtered[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])

    return data_filtered


def test_csp_fractal_feature():
    """
    Testa a combinação CSP + Higuchi Fractal com dados simulados.

    Verifica se a combinação:
    - Executa sem erros
    - Produz output no formato correto
    - CSP melhora a qualidade das features
    """
    print("Testando combinação CSP + Higuchi Fractal...")

    # Dados de teste: 50 trials, 12 canais, 1000 amostras (simulando dados WCCI2020)
    X = np.random.randn(50, 12, 1000)
    y = np.random.randint(0, 2, 50)  # Labels binários

    # Aplica filtro passa-banda (8-30 Hz)
    fs = 250  # Frequência de amostragem WCCI2020
    X_filtered = bandpass_filter(X, lowcut=8, highcut=30, fs=fs)

    # Aplica CSP
    csp_transformer = csp()  # CSP padrão do bciflow
    # CSP do bciflow precisa de X com 4 dimensões: [trials, bands, channels, samples]
    X_4d = X_filtered[:, np.newaxis, :, :]  # Adiciona dimensão de banda
    csp_transformer.fit({"X": X_4d, "y": y})
    X_csp_result = csp_transformer.transform({"X": X_4d})
    X_csp = X_csp_result["X"][
        :, 0, :, :
    ]  # Remove dimensão de banda: [trials, components, samples]

    # Extrai features fractais + estatísticas dos componentes CSP
    # Seguindo o padrão do pipeline oficial: fractal + energy + std por componente
    features = []
    for trial in X_csp:
        # Para cada trial, extraímos os componentes CSP
        comps = (
            trial[:4] if trial.shape[0] >= 4 else trial
        )  # Usando os 4 primeiros componentes
        trial_feat = []
        for comp in comps:
            # 1. Dimensão fractal de Higuchi
            comp_data = {"X": comp.reshape(1, 1, -1)}  # [1 trial, 1 canal, samples]
            fractal_result = higuchi_fractal(comp_data, flating=True)
            fractal_dim = fractal_result["X"][0, 0]  # Extrai valor escalar

            # 2. Energia (log power)
            energy = np.log(np.mean(comp**2) + 1e-10)

            # 3. Desvio padrão
            std_val = np.std(comp)

            # Concatena as 3 features por componente
            trial_feat.extend([fractal_dim, energy, std_val])
        features.append(trial_feat)

    features = np.array(features)  # [n_trials, n_components*3]

    # Verificações de integridade
    assert features.shape[0] == 50, f"Número de trials incorreto: {features.shape[0]}"
    assert (
        features.shape[1] == 12
    ), f"Número de features incorreto: {features.shape[1]} (esperado: 12 = 4 comp × 3 feat)"
    assert not np.isnan(features).any(), "Resultado contém valores NaN"
    assert not np.isinf(features).any(), "Resultado contém valores infinitos"

    # Verifica se as dimensões fractais são positivas (colunas 0, 3, 6, 9)
    fractal_cols = [0, 3, 6, 9]  # Índices das features fractais
    assert np.all(
        features[:, fractal_cols] > 0
    ), "Todas as dimensões fractais devem ser positivas"

    print(f"Shape após CSP + Fractal: {features.shape}")
    print(f"Features por componente: Fractal + Energy + Std = 3")
    print(f"Total de features: {features.shape[1]} (4 componentes × 3 features)")
    print(f"Teste básico concluído com sucesso.")

    return True


def test_csp_fractal_classification_wcci2020():
    """
    Testa classificação usando CSP + Higuchi Fractal em todos os sujeitos do WCCI2020.

    Calcula acurácia e kappa para cada sujeito usando CSP + Fractal + LDA com 5-fold cross-validation.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print("Testando classificação com CSP + Higuchi Fractal no dataset WCCI2020...")
    print("Dataset: WCCI2020 (9 sujeitos)")
    print("Tarefa: Classificação de motor imagery usando CSP + features fractais")
    print(
        "Pipeline: Filtro 8-30Hz -> CSP -> Higuchi Fractal -> Normalização -> LDA (5-fold CV)"
    )
    print("-" * 70)

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

            # Aplica CSP nos dados filtrados
            # CSP do bciflow precisa de X com 4 dimensões: [trials, bands, channels, samples]
            X_4d = X_filtered[:, np.newaxis, :, :]  # Adiciona dimensão de banda
            csp_transformer = csp()  # CSP padrão do bciflow
            csp_transformer.fit({"X": X_4d, "y": y})
            X_csp_result = csp_transformer.transform({"X": X_4d})
            X_csp = X_csp_result["X"][
                :, 0, :, :
            ]  # Remove dimensão de banda: [trials, components, samples]

            # Extrai features fractais + estatísticas dos componentes CSP
            # Seguindo o padrão do pipeline oficial: fractal + energy + std por componente
            features = []
            for trial in X_csp:
                # Para cada trial, extraímos os componentes CSP
                comps = (
                    trial[:4] if trial.shape[0] >= 4 else trial
                )  # Usando os 4 primeiros componentes
                trial_feat = []
                for comp in comps:
                    # 1. Dimensão fractal de Higuchi
                    comp_data = {
                        "X": comp.reshape(1, 1, -1)
                    }  # [1 trial, 1 canal, samples]
                    fractal_result = higuchi_fractal(comp_data, flating=True)
                    fractal_dim = fractal_result["X"][0, 0]  # Extrai valor escalar

                    # 2. Energia (log power)
                    energy = np.log(np.mean(comp**2) + 1e-10)

                    # 3. Desvio padrão
                    std_val = np.std(comp)

                    # Concatena as 3 features por componente
                    trial_feat.extend([fractal_dim, energy, std_val])
                features.append(trial_feat)

            features = np.array(features)  # [n_trials, n_components*3]

            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []

            for train_idx, test_idx in skf.split(features, y):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Normalização por fold para evitar data leakage
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X_train)
                X_test = fold_scaler.transform(X_test)

                # Classificação com LDA
                clf = LDA()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

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
                "n_csp_components": X_csp.shape[1],
                "n_features": features.shape[1],
            }

            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)

            print(
                f"  Acurácia: {mean_accuracy:.4f} | Kappa: {mean_kappa:.4f} | Trials: {X.shape[0]} | CSP: {X_csp.shape[1]} | Features: {features.shape[1]}"
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

    print("Teste de classificação com CSP + Higuchi Fractal concluído com sucesso.")

    return results


if __name__ == "__main__":
    print("=== TESTE: CSP + Higuchi Fractal ===")

    # Teste básico com dados simulados
    print("\n1. Teste básico com dados simulados:")
    success = test_csp_fractal_feature()
    if success:
        print("✓ Teste básico concluído com sucesso.")
    else:
        print("✗ Teste básico falhou.")
        sys.exit(1)

    # Teste com dataset WCCI2020
    print("\n2. Teste de classificação com dataset WCCI2020 (todos os sujeitos):")
    try:
        results = test_csp_fractal_classification_wcci2020()

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
                        "N_CSP_Components": metrics["n_csp_components"],
                        "N_Features": metrics["n_features"],
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                "results/test_outputs/csp_fractal_wcci2020_classification_results.csv",
                index=False,
            )
            print(
                f"\n✓ Resultados salvos em: results/test_outputs/csp_fractal_wcci2020_classification_results.csv"
            )

        print(
            "✓ Teste completo de classificação com CSP + Higuchi Fractal concluído com sucesso!"
        )

    except Exception as e:
        print(f"✗ Erro no teste de classificação com dataset WCCI2020: {str(e)}")
        sys.exit(1)
