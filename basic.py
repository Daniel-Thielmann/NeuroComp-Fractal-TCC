from bcikit.datasets import PhysionetMI
import numpy as np
import matplotlib.pyplot as plt
from bcikit.modules.core import set_global_verbose_eegdata


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


def plot_classes(phy, df_values_all_channels):
    """
    Gera gráficos da Dimensão Fractal média para cada classe (label de 1 a 4).
    """
    labels = phy.labels  # Extrai os labels associados aos trials

    for label in np.unique(labels):  # Itera por cada classe única (1, 2, 3, 4)
        # Filtra os trials pelo label
        df_for_label = df_values_all_channels[labels == label]
        # Calcula a média da DF por canal para essa classe
        df_mean = df_for_label.mean(axis=0)

        # Plot da média da DF por canal
        plt.figure()
        plt.title(f"Dimensão Fractal Média - Classe {label}")
        plt.plot(range(df_mean.shape[0]), df_mean, marker='o',
                 linestyle='-', label=f"DF Média - Classe {label}")
        plt.xlabel("Canais")
        plt.ylabel("Dimensão Fractal")
        plt.legend()
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

    # Plot de um canal específico (trial 19, canal 0) para visualização inicial
    plt.figure()
    plt.title("Sinal EEG Original - Trial 19, Canal 0")
    plt.plot(phy.timestamps, X[19, 0, 0], label="EEG Canal 0")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Corta os dados em uma janela de tempo de 1 a 3 segundos
    cropped_data = phy.crop(tmin=1, window_size=2, inplace=False)
    print("\n--- Após Corte ---")
    print("Formato dos dados cortados:", cropped_data.data.shape)

    # Novo vetor de tempo ajustado
    new_timestamps = np.linspace(1, 1 + 2, cropped_data.data.shape[-1])

    # Plot do sinal cortado
    plt.figure()
    plt.title("Sinal EEG Cortado - Trial 19, Canal 0")
    plt.plot(new_timestamps, cropped_data.data[19, 0, 0], label="EEG Cortado")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # Calcula a Dimensão Fractal (DF) para todos os canais de todos os trials
    df_values_all_channels = []
    for trial_idx in range(cropped_data.data.shape[0]):
        df_trial = []
        # Itera pelos canais
        for channel_idx in range(cropped_data.data.shape[2]):
            signal = cropped_data.data[trial_idx, 0, channel_idx]
            df_value = higuchi_fd(signal)
            df_trial.append(df_value)
        df_values_all_channels.append(df_trial)

    df_values_all_channels = np.array(
        df_values_all_channels)  # Converte para array numpy
    print("\n--- Dimensão Fractal Calculada ---")
    print("Dimensão Fractal para todos os canais (formato [trials, canais]):")
    print(df_values_all_channels)

    # Gera os gráficos para cada classe (label)
    plot_classes(phy, df_values_all_channels)

    # Gera o histograma ordenado dos valores de DF
    plot_sorted_histogram(df_values_all_channels)


if __name__ == "__main__":
    main()
