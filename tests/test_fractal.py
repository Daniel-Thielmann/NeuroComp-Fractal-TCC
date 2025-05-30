import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Adiciona o diretório raiz ao path do Python para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bciflow.datasets import cbcic
from methods.features.fractal import HiguchiFractalEvolution


def test_higuchi_fractal_evolution_init():
    """Testa a inicialização da classe HiguchiFractalEvolution com parâmetros padrão e personalizados."""
    # Testa inicialização com parâmetros padrão
    hfd = HiguchiFractalEvolution()
    assert hfd.kmax == 100, f"Valor padrão incorreto para kmax: {hfd.kmax}"
    assert len(hfd.bands) == 5, f"Número incorreto de bandas: {len(hfd.bands)}"
    assert hfd.sfreq == 512, f"Valor padrão incorreto para sfreq: {hfd.sfreq}"

    # Testa inicialização com parâmetros personalizados
    custom_bands = [
        ("alpha", 8, 13),
        ("beta", 13, 30),
    ]
    hfd_custom = HiguchiFractalEvolution(kmax=50, bands=custom_bands, sfreq=250)
    assert (
        hfd_custom.kmax == 50
    ), f"Valor personalizado incorreto para kmax: {hfd_custom.kmax}"
    assert (
        len(hfd_custom.bands) == 2
    ), f"Número incorreto de bandas personalizadas: {len(hfd_custom.bands)}"
    assert (
        hfd_custom.sfreq == 250
    ), f"Valor personalizado incorreto para sfreq: {hfd_custom.sfreq}"


def test_higuchi_fractal_evolution_create_filter_bank():
    """Testa a criação do banco de filtros."""
    hfd = HiguchiFractalEvolution()
    filter_bank = hfd.filter_bank

    # Verifica se foram criados filtros para todas as bandas
    assert len(filter_bank) == len(
        hfd.bands
    ), "Número de filtros diferente do número de bandas"

    # Verifica se os filtros têm a estrutura correta (b, a)
    for band_name, (b, a) in filter_bank.items():
        assert isinstance(
            b, np.ndarray
        ), f"Coeficientes b para {band_name} não são numpy array"
        assert isinstance(
            a, np.ndarray
        ), f"Coeficientes a para {band_name} não são numpy array"


def test_higuchi_fractal_evolution_calculate_enhanced_hfd():
    """Testa o cálculo da dimensão fractal de Higuchi em sinais sintéticos."""
    hfd = HiguchiFractalEvolution()

    # Teste com sinal senoidal (dimensão esperada próxima a 1)
    t = np.linspace(0, 1, 512)
    sine_wave = np.sin(2 * np.pi * 10 * t)
    slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(sine_wave)
    assert 0.9 < slope < 1.3, f"Dimensão fractal inesperada para onda senoidal: {slope}"

    # Teste com ruído branco (dimensão esperada próxima a 1-2)
    white_noise = np.random.randn(512)
    slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(white_noise)
    assert 1.0 < slope < 2.2, f"Dimensão fractal inesperada para ruído branco: {slope}"

    # Teste com sinal pequeno (deve retornar zeros)
    small_signal = np.array([1, 2, 3])
    slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(small_signal)
    assert (
        slope == 0.0 and mean_lk == 0.0 and std_lk == 0.0
    ), "Falha no tratamento de sinal pequeno"


def test_higuchi_fractal_evolution_extract():
    """Testa a extração de características fractais de sinais EEG simulados."""
    hfd = HiguchiFractalEvolution()

    # Cria dados de EEG simulados (3 trials, 4 canais, 512 amostras)
    n_trials, n_channels, n_samples = 3, 4, 512
    X = np.random.randn(n_trials, n_channels, n_samples)

    # Extrai características
    features = hfd.extract(X)

    # Verifica dimensões corretas (n_trials x (n_channels * n_bands * 5))
    expected_features = n_trials, n_channels * len(hfd.bands) * 5
    assert (
        features.shape == expected_features
    ), f"Shape incorreto: {features.shape}, esperado: {expected_features}"

    # Verifica se não há valores NaN
    assert not np.isnan(features).any(), "Características contêm valores NaN"


def test_fractal_real_data():
    """Testa o cálculo de características fractais em dados reais de EEG."""
    try:
        # Configurar seed para reprodutibilidade
        np.random.seed(42)

        # Carrega dados de um sujeito para teste
        subject_id = 1
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"].squeeze(1)
        y = np.array(dataset["y"]) + 1

        # Filtra classes 1 e 2
        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        # Pré-processamento melhorado
        # 1. Centraliza o sinal (remoção de offset DC)
        X = X - np.mean(X, axis=2, keepdims=True)
        # 2. Normalização da amplitude
        X = X / (np.std(X, axis=2, keepdims=True) + 1e-10)

        # Inicializa o extrator de características com kmax otimizado
        hfd = HiguchiFractalEvolution(kmax=150)

        # Extrai características
        features = hfd.extract(X)

        # Verifica dimensões corretas e ausência de NaNs
        assert (
            features.shape[0] == X.shape[0]
        ), "Número incorreto de exemplos nas características"
        assert not np.isnan(features).any(), "Características contêm valores NaN"

        # Normalização robusta
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # PCA com preservação de variância
        pca = PCA(n_components=0.98)  # Preserva 98% da variância
        X_pca = pca.fit_transform(X_scaled)

        # Validação cruzada mais robusta
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, test_idx in skf.split(X_pca, y):
            clf = LDA(solver="svd")  # Solucionar numérico mais estável
            clf.fit(X_pca[train_idx], y[train_idx])
            acc = clf.score(X_pca[test_idx], y[test_idx])
            accuracies.append(acc)

        # Calcular acurácia média
        mean_accuracy = np.mean(accuracies)

        # Verificar se a acurácia média está acima do nível de chance (50%)
        assert (
            mean_accuracy > 0.5
        ), f"Acurácia abaixo do nível de chance: {mean_accuracy}"
        print(
            f"Acurácia média na classificação com características fractais: {mean_accuracy:.4f}"
        )

    except Exception as e:
        assert False, f"Erro ao testar com dados reais: {str(e)}"


def run_fractal_test():
    """Executa o teste completo do método fractal em todos os sujeitos."""
    all_rows = []
    # Otimização de parâmetros - kmax maior captura mais detalhes fractais
    hfd = HiguchiFractalEvolution(
        kmax=150
    )  # Aumentando kmax para capturar mais detalhes

    # Usar seed para garantir reprodutibilidade
    np.random.seed(42)

    for subject_id in tqdm(range(1, 10), desc="Fractal"):
        dataset = cbcic(subject=subject_id, path="dataset/wcci2020/")
        X = dataset["X"].squeeze(1)
        y = np.array(dataset["y"]) + 1

        mask = (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        # Pré-processamento melhorado
        # 1. Centraliza o sinal por canal/trial (remoção de offset DC)
        X = X - np.mean(X, axis=2, keepdims=True)

        # 2. Normalização por trial para reduzir variabilidade entre sessões
        # Dividir pelo desvio padrão de cada trial para normalizar amplitude
        X = X / (np.std(X, axis=2, keepdims=True) + 1e-10)

        # Extração de características com parâmetros otimizados
        features = hfd.extract(X)

        # Normalização robusta com clipping para remover outliers
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Ajustando PCA para preservar mais variância (usando variance_threshold)
        variance_threshold = 0.98  # Preservar 98% da variância
        pca = PCA(n_components=variance_threshold)
        features = pca.fit_transform(features)

        # Aumentando robustez da validação cruzada
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        subject_rows = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
            clf = LDA(solver="svd")  # Usar solver SVD para maior estabilidade numérica
            clf.fit(features[train_idx], y[train_idx])
            probs = clf.predict_proba(features[test_idx])

            # Coeficientes da LDA podem indicar características importantes
            if fold_idx == 0:  # Salvar apenas uma vez por sujeito
                coef_file = os.path.join(
                    "results/Fractal/Training", f"P{subject_id:02d}_coefs.csv"
                )
                pd.DataFrame(
                    {
                        "feature_idx": np.arange(features.shape[1]),
                        "coefficient": clf.coef_[0],
                    }
                ).to_csv(coef_file, index=False)

            for i, idx in enumerate(test_idx):
                row = {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y[idx],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                    "predicted_label": 1 if probs[i][0] > probs[i][1] else 2,
                }
                subject_rows.append(row)
                all_rows.append(row)

        # Calcular e salvar acurácia por sujeito
        subject_df = pd.DataFrame(subject_rows)
        subject_acc = (subject_df["true_label"] == subject_df["predicted_label"]).mean()
        print(
            f"Sujeito {subject_id}: Acurácia = {subject_acc:.4f} ({len(subject_df)} amostras)"
        )

        # Salva resultados por sujeito
        os.makedirs("results/Fractal/Training", exist_ok=True)
        subject_df.to_csv(
            f"results/Fractal/Training/P{subject_id:02d}.csv", index=False
        )

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # Executa os testes unitários
    test_higuchi_fractal_evolution_init()
    test_higuchi_fractal_evolution_create_filter_bank()
    test_higuchi_fractal_evolution_calculate_enhanced_hfd()
    test_higuchi_fractal_evolution_extract()
    print("Todos os testes unitários passaram!")

    # Executa o teste com dados reais
    test_fractal_real_data()
    print("Teste com dados reais passou!")

    # Executa o teste completo e exibe resultados
    df = run_fractal_test()  # Calcula métricas de desempenho
    df["correct_prob"] = df.apply(
        lambda row: row["left_prob"] if row["true_label"] == 1 else row["right_prob"],
        axis=1,
    )

    # Cálculo de acurácia geral
    acc = (df["true_label"] == df["predicted_label"]).mean()

    # Média de probabilidade correta (confiança do modelo)
    mean_prob = df["correct_prob"].mean()

    # Estatísticas gerais
    total = len(df)
    counts = dict(df["true_label"].value_counts().sort_index())

    # Acurácia por sujeito
    subject_acc = df.groupby("subject_id").apply(
        lambda x: (x["true_label"] == x["predicted_label"]).mean()
    )

    # Salva sumário de resultados
    summary_file = "results/Fractal/summary.csv"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    subject_acc.to_csv(summary_file)

    # Exibe resultados
    print(f"Acurácia global: {acc:.4f}")
    print(f"Média Prob. Correta: {mean_prob:.4f}")
    print(f"Total amostras: {total} | Distribuição de classes: {counts}")
    print("\nAcurácia por sujeito:")
    for subject_id, acc_value in subject_acc.items():
        print(f"Sujeito {subject_id}: {acc_value:.4f}")
