from bcikit.datasets import PhysionetMI
import numpy as np
import matplotlib.pyplot as plt
from bcikit.modules.core import set_global_verbose_eegdata
import time


def higuchi_fd(signal, kmax=10):
    """
    Calcula a Dimensão Fractal de Higuchi para um sinal EEG.
    Higuchi DF mede a complexidade de um sinal em diferentes escalas.
    """
    L = []  # Lista para armazenar os comprimentos calculados
    x = np.asarray(signal)  # Converte o sinal em um array numpy
    N = x.size  # Tamanho do sinal

    # Loop para diferentes valores de 'k' (tamanho da subsequência)
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(0, k):  # Offset inicial
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                # Diferença entre pontos
                Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
            Lmk /= k * ((N - 1) / k)  # Normaliza pelo tamanho da subsequência
            Lk += Lmk
        # Calcula logaritmo do comprimento normalizado
        L.append(np.log(Lk / k))

    # Regressão linear no gráfico log-log para obter a inclinação da reta (DF)
    return -np.polyfit(np.log(range(1, kmax + 1)), L, 1)[0]


def euclidean_alignment(data):
    """
    Implementa o alinhamento euclidiano nos dados EEG.

    Parâmetros:
    - data: numpy array de formato (trials, canais, amostras), onde:
      - trials: número de ensaios
      - canais: número de canais EEG
      - amostras: número de pontos temporais por canal

    Retorno:
    - data_aligned: numpy array com os dados alinhados
    """
    start_time = time.time()

    # Remove o eixo singleton (trials, 1, canais, amostras) -> (trials, canais, amostras)
    data = np.squeeze(data, axis=1)

    # Reorganiza os dados para (trials, amostras, canais) e calcula a covariância
    # (trials, amostras, canais)
    data_reordered = np.transpose(data, (0, 2, 1))

    # Calcula a matriz de covariância média entre os canais
    covariance_mean = np.mean([
        np.cov(trial, rowvar=False) for trial in data_reordered
    ], axis=0)

    # Decomposição espectral (autovalores e autovetores) da matriz de covariância média
    eigvals, eigvecs = np.linalg.eigh(covariance_mean)

    # Matriz de transformação para o alinhamento
    whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Aplica a transformação em cada trial
    data_aligned = np.array([
        trial @ whitening_matrix for trial in data_reordered
    ])

    # Retorna ao formato original (trials, canais, amostras)
    data_aligned = np.transpose(data_aligned, (0, 2, 1))

    print(f"Tempo de alinhamento euclidiano: {
          time.time() - start_time:.2f} segundos")
    return data_aligned


def plot_classes(phy, df_values_all_channels):
    """
    Gera gráficos da Dimensão Fractal média para cada classe (label de 1 a 4).
    """
    labels = phy.labels  # Extrai os labels associados aos trials
    plt.figure()

    for label in np.unique(labels):  # Itera por cada classe única (1, 2, 3, 4)
        # Filtra os trials pelo label
        df_for_label = df_values_all_channels[labels == label]
        # Calcula a média da DF por canal para essa classe
        df_mean = df_for_label.mean(axis=0)

        # Plot da média da DF por canal
        plt.plot(range(df_mean.shape[0]), df_mean, marker='o',
                 linestyle='-', label=f"DF Média - Label {label}")

    plt.xlabel("Canais")
    plt.ylabel("Dimensão Fractal Média")
    plt.legend()
    plt.title("Dimensão Fractal por Classe")
    plt.show()


def plot_sorted_histogram(df_values_all_channels):
    """
    Gera um histograma dos valores de DF ordenados do menor para o maior.
    """
    df_all = df_values_all_channels.flatten(
    )  # Achata a matriz para pegar todos os valores de DF
    df_sorted = np.sort(df_all)  # Ordena os valores de DF em ordem crescente

    # Plot do histograma
    plt.figure()
    plt.title("Histograma dos Valores de DF Ordenados")
    plt.hist(df_sorted, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Dimensão Fractal")
    plt.ylabel("Frequência")
    plt.show()


def compare_before_after(aligned_df, original_df):
    """
    Compara os valores de dimensão fractal antes e depois do alinhamento euclidiano.
    """
    plt.figure()
    plt.hist(original_df.flatten(), bins=30, color='blue',
             alpha=0.5, label='DF Original', edgecolor='black')
    plt.hist(aligned_df.flatten(), bins=30, color='green',
             alpha=0.5, label='DF Alinhado', edgecolor='black')
    plt.title("Comparação DF Antes e Depois do Alinhamento")
    plt.xlabel("Dimensão Fractal")
    plt.ylabel("Frequência")
    plt.legend()
    plt.show()


def main():
    # Configura o nível de log
    set_global_verbose_eegdata("WARNING")

    # Carrega o dataset Physionet Motor Imagery
    phy = PhysionetMI.loading(verbose="WARNING")
    print("\n--- Informações Iniciais ---")
    print("Formato dos dados:", phy.data.shape)  # Mostra o formato dos dados
    # Mostra os labels das tarefas motoras
    print("Labels disponíveis:", phy.labels)

    X = phy.data  # Extrai os dados EEG

    # Calcula a Dimensão Fractal para os dados originais
    original_df_values = []
    for trial_idx in range(X.shape[0]):
        df_trial = []
        for channel_idx in range(X.shape[2]):
            signal = X[trial_idx, 0, channel_idx]
            df_value = higuchi_fd(signal)
            df_trial.append(df_value)
        original_df_values.append(df_trial)
    original_df_values = np.array(original_df_values)

    # Gera os gráficos para os dados originais
    print("\n--- Gerando gráficos para os dados originais ---")
    plot_classes(phy, original_df_values)
    plot_sorted_histogram(original_df_values)

    # Aplica o alinhamento euclidiano nos dados
    aligned_data = euclidean_alignment(X)
    print("\n--- Dados Alinhados ---")
    print("Formato dos dados alinhados:", aligned_data.shape)

    # Calcula a Dimensão Fractal para os dados alinhados
    aligned_df_values = []
    for trial_idx in range(aligned_data.shape[0]):
        df_trial = []
        # Ajuste aqui para usar o eixo correto
        for channel_idx in range(aligned_data.shape[1]):
            signal = aligned_data[trial_idx, channel_idx, :]
            df_value = higuchi_fd(signal)
            df_trial.append(df_value)
        aligned_df_values.append(df_trial)
    aligned_df_values = np.array(aligned_df_values)

    # Gera os gráficos para os dados alinhados
    print("\n--- Gerando gráficos para os dados alinhados ---")
    plot_classes(phy, aligned_df_values)
    plot_sorted_histogram(aligned_df_values)

    # Compara os valores antes e depois do alinhamento
    compare_before_after(aligned_df_values, original_df_values)


if __name__ == "__main__":
    main()
