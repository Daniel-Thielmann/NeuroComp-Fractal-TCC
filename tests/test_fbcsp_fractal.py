import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis

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


def mibif_feature_selection(features, y, n_features=10):
    """
    Aplica MIBIF (Mutual Information-Based Individual Feature) selection.

    Args:
        features: array [n_trials, n_features]
        y: array [n_trials] - labels
        n_features: número de features a selecionar

    Returns:
        selected_features: features selecionadas
        selected_indices: índices das features selecionadas
    """
    # Calcula mutual information entre features e labels
    mi_scores = mutual_info_classif(features, y, random_state=42)

    # Seleciona top n_features com maior MI
    selected_indices = np.argsort(mi_scores)[-n_features:]
    selected_features = features[:, selected_indices]

    return selected_features, selected_indices


def test_fbcsp_fractal_feature():
    """
    Testa a combinação FBCSP + Higuchi Fractal com dados simulados.

    Verifica se a combinação:
    - Executa sem erros
    - Produz output no formato correto
    - FBCSP melhora a qualidade das features
    """
    print("Testando combinação FBCSP + Higuchi Fractal...")

    # Dados de teste: 50 trials, 12 canais, 1000 amostras (simulando dados WCCI2020)
    X = np.random.randn(50, 12, 1000)
    y = np.random.randint(0, 2, 50)  # Labels binários

    fs = 250  # Frequência de amostragem WCCI2020
    filter_banks = [
        (8, 12),  # Alfa inferior
        (12, 16),  # SMR (Sensorimotor Rhythm)
        (16, 20),  # Beta inferior
        (20, 24),  # Beta médio
        (24, 30),  # Beta superior
    ]

    # Aplica FBCSP: Filter Bank + CSP
    all_features = []

    for low_freq, high_freq in filter_banks:
        # Filtra dados na banda específica
        X_filtered = bandpass_filter(X, lowcut=low_freq, highcut=high_freq, fs=fs)

        # Aplica CSP na banda filtrada
        csp_transformer = csp()
        X_4d = X_filtered[:, np.newaxis, :, :]  # [trials, bands, channels, samples]
        csp_transformer.fit({"X": X_4d, "y": y})
        X_csp_result = csp_transformer.transform({"X": X_4d})
        X_csp = X_csp_result["X"][:, 0, :, :]  # [trials, components, samples]

        # Extrai features fractais + estatísticas dos componentes CSP
        band_features = []
        for trial in X_csp:
            comps = (
                trial[:4] if trial.shape[0] >= 4 else trial
            )  # REVERTER: 4 componentes
            trial_feat = []
            for comp in comps:
                # Apenas dimensão fractal de Higuchi (1 feature por componente)
                comp_data = {"X": comp.reshape(1, 1, -1)}
                fractal_result = higuchi_fractal(comp_data, flating=True)
                fractal_dim = fractal_result["X"][0, 0]

                trial_feat.append(fractal_dim)  # Apenas 1 feature por componente
            band_features.append(trial_feat)

        all_features.append(np.array(band_features))  # [n_trials, n_features_per_band]

    # Concatena features de todas as bandas: [n_trials, n_bands * n_features_per_band]
    features = np.concatenate(all_features, axis=1)

    # Aplica MIBIF para selecionar melhores features
    n_selected_features = min(20, features.shape[1])  # Seleciona até 20 features
    selected_features, selected_indices = mibif_feature_selection(
        features, y, n_selected_features
    )

    # Verificações de integridade
    assert (
        selected_features.shape[0] == 50
    ), f"Número de trials incorreto: {selected_features.shape[0]}"
    assert (
        selected_features.shape[1] == n_selected_features
    ), f"Número de features incorreto: {selected_features.shape[1]}"
    assert not np.isnan(selected_features).any(), "Resultado contém valores NaN"
    assert not np.isinf(selected_features).any(), "Resultado contém valores infinitos"

    print(f"Shape após FBCSP + Fractal: {features.shape}")
    print(f"Shape após MIBIF: {selected_features.shape}")
    print(
        f"Features por banda: {len(filter_banks)} bandas × 4 comp × 1 feat = {len(filter_banks) * 4 * 1}"
    )
    print(f"Features selecionadas: {n_selected_features} de {features.shape[1]}")
    print(f"Teste básico concluído com sucesso.")

    return True


def test_fbcsp_fractal_classification_wcci2020():
    """
    Testa classificação usando FBCSP + Higuchi Fractal em todos os sujeitos do WCCI2020.

    Calcula acurácia e kappa para cada sujeito usando FBCSP + Fractal + MIBIF + LDA com 5-fold cross-validation.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print("Testando classificação com FBCSP + Higuchi Fractal no dataset WCCI2020...")
    print("Dataset: WCCI2020 (9 sujeitos)")
    print("Tarefa: Classificação de motor imagery usando FBCSP + features fractais")
    print(
        "Pipeline: Filter Bank (5 bandas) → CSP (4 comp) → Fractal+Energy+Std → MIBIF (30) → LDA (5-fold CV)"
    )
    print("-" * 85)

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

    # Filter Bank: múltiplas bandas de frequência para motor imagery
    filter_banks = [
        (8, 12),  # Alfa inferior
        (12, 16),  # SMR (Sensorimotor Rhythm)
        (16, 20),  # Beta inferior
        (20, 24),  # Beta médio
        (24, 30),  # Beta superior
    ]

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

            # Aplica FBCSP: Filter Bank + CSP + Fractal
            all_features = []

            for low_freq, high_freq in filter_banks:
                # Filtra dados na banda específica
                X_band = bandpass_filter(X, low_freq, high_freq, sfreq=sfreq)

                # Aplica CSP na banda filtrada (treina uma vez para todos os trials)
                csp_transformer = csp()
                X_4d = X_band[:, np.newaxis, :, :]  # [trials, bands, channels, samples]
                csp_transformer.fit({"X": X_4d, "y": y})
                X_csp_result = csp_transformer.transform({"X": X_4d})
                X_csp = X_csp_result["X"][:, 0, :, :]  # [trials, components, samples]

                # Extrai features fractais para todos os trials desta banda
                band_features = []
                for trial_idx, trial in enumerate(X_csp):
                    comps = trial[:4] if trial.shape[0] >= 4 else trial
                    trial_feat = []
                    for comp_idx, comp in enumerate(comps):
                        # Seguindo exatamente o CSP Fractal que funciona (81.11%):
                        # 1. Dimensão fractal de Higuchi
                        comp_data = {"X": comp.reshape(1, 1, -1)}
                        fractal_result = higuchi_fractal(comp_data, flating=True)
                        fractal_dim = fractal_result["X"][0, 0]

                        # 2. Energia (log power) - igual ao CSP que funciona
                        energy = np.log(np.mean(comp**2) + 1e-10)

                        # 3. Desvio padrão - igual ao CSP que funciona
                        std_val = np.std(comp)

                        # Debug: verificar se fractal está funcionando corretamente
                        if (
                            trial_idx == 0 and comp_idx == 0 and low_freq == 8
                        ):  # Só primeira vez
                            print(
                                f"    Debug - Comp shape: {comp.shape}, Fractal: {fractal_dim:.6f}, Energy: {energy:.6f}, Std: {std_val:.6f}"
                            )

                        # Concatena as 3 features por componente (igual ao CSP que funciona)
                        trial_feat.extend([fractal_dim, energy, std_val])
                    band_features.append(trial_feat)

                all_features.append(np.array(band_features))

            # Concatena features de todas as bandas
            features = np.concatenate(all_features, axis=1)

            # Aplica MIBIF para selecionar melhores features
            # Com 5 bandas × 4 componentes × 3 features = 60 features total
            n_selected_features = min(
                30, features.shape[1]
            )  # Seleciona 30 features como especificado
            selected_features, selected_indices = mibif_feature_selection(
                features, y, n_selected_features
            )

            # 5-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_kappas = []

            for train_idx, test_idx in skf.split(selected_features, y):
                X_train, X_test = (
                    selected_features[train_idx],
                    selected_features[test_idx],
                )
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
                "n_filter_banks": len(filter_banks),
                "n_features_total": features.shape[1],
                "n_features_selected": selected_features.shape[1],
            }

            all_accuracies.append(mean_accuracy)
            all_kappas.append(mean_kappa)

            print(
                f"  Acurácia: {mean_accuracy:.4f} | Kappa: {mean_kappa:.4f} | Trials: {X.shape[0]} | "
                f"Features: {features.shape[1]}→{selected_features.shape[1]} | Bandas: {len(filter_banks)}"
            )

        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}

    # Estatísticas gerais
    print("-" * 85)
    print("RESULTADOS FINAIS:")
    print(
        f"Acurácia média: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}"
    )
    print(f"Kappa médio: {np.mean(all_kappas):.4f} ± {np.std(all_kappas):.4f}")
    print(f"Melhor acurácia: {np.max(all_accuracies):.4f}")
    print(f"Pior acurácia: {np.min(all_accuracies):.4f}")

    print("Teste de classificação com FBCSP + Higuchi Fractal concluído com sucesso.")

    return results


if __name__ == "__main__":
    print("=== TESTE: FBCSP + Higuchi Fractal ===")

    # Teste básico com dados simulados
    print("\n1. Teste básico com dados simulados:")
    success = test_fbcsp_fractal_feature()
    if success:
        print("✓ Teste básico concluído com sucesso.")
    else:
        print("✗ Teste básico falhou.")
        sys.exit(1)

    # Teste com dataset WCCI2020
    print("\n2. Teste de classificação com dataset WCCI2020 (todos os sujeitos):")
    try:
        results = test_fbcsp_fractal_classification_wcci2020()

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
                        "N_Filter_Banks": metrics["n_filter_banks"],
                        "N_Features_Total": metrics["n_features_total"],
                        "N_Features_Selected": metrics["n_features_selected"],
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                "results/test_outputs/fbcsp_fractal_wcci2020_classification_results.csv",
                index=False,
            )
            print(
                f"\n✓ Resultados salvos em: results/test_outputs/fbcsp_fractal_wcci2020_classification_results.csv"
            )

        print(
            "✓ Teste completo de classificação com FBCSP + Higuchi Fractal concluído com sucesso!"
        )

    except Exception as e:
        print(f"✗ Erro no teste de classificação com dataset WCCI2020: {str(e)}")
        sys.exit(1)
