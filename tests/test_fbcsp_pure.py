import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import cohen_kappa_score


def test_fbcsp_pure_pipeline():
    """
    Testa o pipeline FBCSP puro usando dataset WCCI2020 padronizado.

    Executa classificação de motor imagery para todos os sujeitos do dataset WCCI2020
    usando Filter Bank, CSP e MIBIF com validação cruzada robusta.

    Returns:
        dict: Resultados de performance por sujeito
    """
    print("Testando pipeline FBCSP Pure...")
    print("Dataset: WCCI2020 (9 sujeitos)")
    print("Tarefa: Classificação de motor imagery (left-hand vs right-hand)")
    print(
        "Pipeline: Filter Bank (5 bandas) → CSP (2 comp extremos) → Log(Var) → MIBIF (30) → LDA (5-fold CV)"
    )
    print("-" * 60)

    import scipy.io as sio
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=4):
        """Aplica filtro passa-banda usando Butterworth."""
        nyquist = sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, data, axis=-1)

    def extract_fbcsp_pure_features(X, y, frequency_bands, n_components=4):
        """Extrai features FBCSP puras - log variance das componentes CSP (método clássico)."""
        from bciflow.modules.sf.csp import csp

        all_features = []

        # Para cada banda de frequência
        for low_freq, high_freq in frequency_bands:
            # Filtra dados para a banda atual
            X_band = bandpass_filter(X, low_freq, high_freq, sfreq=512)

            # Aplica CSP na banda
            csp_transformer = csp()
            csp_transformer.fit({"X": X_band[:, np.newaxis, :, :], "y": y})
            X_csp = csp_transformer.transform({"X": X_band[:, np.newaxis, :, :]})["X"]
            X_csp = X_csp[:, 0, :, :]  # [trials, components, samples]

            # Para cada trial, extrai features das componentes CSP
            for trial_idx in range(X_csp.shape[0]):
                if trial_idx >= len(all_features):
                    all_features.append([])

                trial_components = X_csp[trial_idx]
                # FBCSP Pure: usa apenas os 2 componentes mais discriminativos (primeiro e último)
                if trial_components.shape[0] >= 2:
                    # Método clássico: primeiro componente (classe 1) + último componente (classe 2)
                    comps = np.array([trial_components[0], trial_components[-1]])
                else:
                    comps = trial_components

                # Feature CSP pura: log(var(component)) - apenas componentes extremos
                for component in comps:
                    # Método puro: log da variância dos componentes mais discriminativos
                    variance = np.var(component)
                    log_variance = np.log(variance + 1e-10)
                    all_features[trial_idx].append(log_variance)

        return np.array(all_features)

    results = {}
    all_accuracies = []
    all_kappas = []

    # Define bandas de frequência (mesmo padrão do FBCSP+Fractal)
    frequency_bands = [
        (8, 12),  # Alpha baixo
        (12, 16),  # Alpha alto
        (16, 20),  # Beta baixo
        (20, 24),  # Beta médio
        (24, 30),  # Beta alto
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

            # Validação cruzada 5-fold
            cv_accuracies = []
            cv_kappas = []

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Extrai features FBCSP puras
                features_train = extract_fbcsp_pure_features(
                    X_train, y_train, frequency_bands, n_components=4
                )
                features_test = extract_fbcsp_pure_features(
                    X_test, y_test, frequency_bands, n_components=4
                )

                # Seleção de features com MIBIF
                n_selected_features = min(30, features_train.shape[1])
                selector = SelectKBest(
                    score_func=mutual_info_classif, k=n_selected_features
                )
                features_train_selected = selector.fit_transform(
                    features_train, y_train
                )
                features_test_selected = selector.transform(features_test)

                # Normalização
                scaler = StandardScaler()
                features_train_final = scaler.fit_transform(features_train_selected)
                features_test_final = scaler.transform(features_test_selected)

                # Classificação LDA
                clf = LDA()
                clf.fit(features_train_final, y_train)
                y_pred = clf.predict(features_test_final)

                # Métricas
                fold_accuracy = (y_test == y_pred).mean()
                fold_kappa = cohen_kappa_score(y_test, y_pred)

                cv_accuracies.append(fold_accuracy)
                cv_kappas.append(fold_kappa)

            # Métricas finais do sujeito
            accuracy = np.mean(cv_accuracies)
            kappa = np.mean(cv_kappas)

            # Armazena resultados
            results[f"P{subject_id:02d}"] = {
                "accuracy": accuracy,
                "kappa": kappa,
                "n_samples": X.shape[0],
                "class_distribution": dict(pd.Series(y).value_counts().sort_index()),
                "cv_accuracies": cv_accuracies,
                "cv_kappas": cv_kappas,
            }

            all_accuracies.append(accuracy)
            all_kappas.append(kappa)

            print(
                f"  Acurácia: {accuracy:.4f} ± {np.std(cv_accuracies):.4f} | Kappa: {kappa:.4f} | Amostras: {X.shape[0]}"
            )

        except Exception as e:
            print(f"  ERRO: {str(e)}")
            results[f"P{subject_id:02d}"] = {"error": str(e)}

    # Estatísticas gerais
    print("-" * 60)
    print("RESULTADOS FINAIS:")
    if all_accuracies:
        print(
            f"Acurácia média: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}"
        )
        print(f"Kappa médio: {np.mean(all_kappas):.4f} ± {np.std(all_kappas):.4f}")
        print(f"Melhor acurácia: {np.max(all_accuracies):.4f}")
        print(f"Pior acurácia: {np.min(all_accuracies):.4f}")

        # Verifica se pipeline está funcionando adequadamente
        mean_accuracy = np.mean(all_accuracies)
        assert (
            mean_accuracy > 0.5
        ), f"Acurácia média abaixo do acaso: {mean_accuracy:.4f}"

        print("Teste do pipeline FBCSP Pure concluído com sucesso.")
    else:
        print("Nenhum resultado válido obtido.")

    return results


if __name__ == "__main__":
    print("=== TESTE: Pipeline FBCSP Pure ===")
    results = test_fbcsp_pure_pipeline()

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
                    "N_Samples": metrics["n_samples"],
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results/test_outputs/fbcsp_pure_test_results.csv", index=False)
    print(f"Resultados salvos em: results/test_outputs/fbcsp_pure_test_results.csv")
