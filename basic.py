from bcikit.datasets import PhysionetMI
import numpy as np
import matplotlib.pyplot as plt
from bcikit.modules.core import set_global_verbose_eegdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Lista padrão de nomes de eletrodos para 64 canais (definida como variável global)
electrode_names = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP7", "TP8",
    "AF3", "AF4", "F1", "F2", "C1", "C2", "P1", "P2", "PO3", "PO4", "F5", "F6", "C5", "C6", "P5", "P6",
    "PO7", "PO8", "FT7", "FT8", "TP9", "TP10", "Fpz", "CPz", "AF7", "AF8", "F9", "F10", "T9", "T10", "P9", "P10",
    "Iz", "Oz2"
]

# Garantindo que a lista de eletrodos tenha 64 nomes
assert len(electrode_names) == 64, "A lista de nomes de eletrodos precisa conter exatamente 64 elementos."


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

    return data_aligned


def evaluate_accuracy(df_values_all_channels, labels):
    """
    Avalia a acurácia de classificação usando LDA nos labels 1 e 2.
    """
    # Filtra apenas os labels 1 e 2
    mask = (labels == 1) | (labels == 2)
    X = df_values_all_channels[mask]
    y = labels[mask]

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Instancia e treina o modelo LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Faz previsões e calcula a acurácia
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Acurácia nos labels 1 e 2: {accuracy:.2f}%")


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

        # Ajusta a lista de eletrodos para o número de canais disponíveis
        electrodes_used = electrode_names[:df_mean.shape[0]]

        # Plot da média da DF por canal
        plt.plot(electrodes_used, df_mean, marker='o',
                 linestyle='-', label=f"DF Média - Label {label}")

    plt.xlabel("Eletrodos")
    plt.ylabel("Dimensão Fractal Média")
    plt.legend()
    plt.title("Dimensão Fractal por Classe")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_sorted_histogram(df_values_all_channels):
    """
    Gera um histograma dos valores de DF ordenados por eletrodos.
    """
    df_mean_per_channel = df_values_all_channels.mean(
        axis=0)  # Média por canal

    # Ajusta a lista de eletrodos para o número de canais disponíveis
    electrodes_used = electrode_names[:df_mean_per_channel.shape[0]]

    # Plot do histograma
    plt.figure()
    plt.bar(electrodes_used, df_mean_per_channel,
            color='skyblue', edgecolor='black', width=0.5)
    plt.title("Histograma da Dimensão Fractal por Eletrodo")
    plt.xlabel("Eletrodos")
    plt.ylabel("Dimensão Fractal Média")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def compare_before_after(aligned_df, original_df):
    """
    Compara os valores de dimensão fractal antes e depois do alinhamento euclidiano.
    """
    aligned_mean = aligned_df.mean(axis=0)  # Média dos valores alinhados
    original_mean = original_df.mean(axis=0)  # Média dos valores originais
    df_diff = aligned_mean - original_mean  # Diferença média por canal

    # Ajusta a lista de eletrodos para o número de canais disponíveis
    electrodes_used = electrode_names[:df_diff.shape[0]]

    x = np.arange(len(electrodes_used))  # Posições dos rótulos no eixo x

    plt.figure(figsize=(12, 6))
    width = 0.25  # Largura das barras

    # Barras dos valores originais
    plt.bar(x - width, original_mean, width=width, color='blue',
            edgecolor='black', label='DF Original')

    # Barras dos valores alinhados
    plt.bar(x, aligned_mean, width=width, color='green',
            edgecolor='black', label='DF Alinhado')

    # Barras das diferenças
    plt.bar(x + width, df_diff, width=width, color='orange',
            edgecolor='black', label='Diferença (Alinhado - Original)')

    plt.xticks(x, electrodes_used, rotation=90)
    plt.xlabel("Eletrodos")
    plt.ylabel("Dimensão Fractal Média")
    plt.title("Comparação DF Antes, Depois e Diferença por Eletrodo")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lda_qda_comparison(X, y):
    """
    Gera um gráfico comparativo entre LDA e QDA usando os labels 1 e 2.
    """
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Ajuste do LDA e QDA
    lda.fit(X_train, y_train)
    qda.fit(X_train, y_train)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Previsões para o LDA
    Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lda = Z_lda.reshape(xx.shape)

    # Previsões para o QDA
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_qda = Z_qda.reshape(xx.shape)

    plt.figure(figsize=(12, 6))

    # Plot do LDA
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z_lda, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Linear Discriminant Analysis")

    # Plot do QDA
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z_qda, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Quadratic Discriminant Analysis")

    plt.tight_layout()
    plt.show()


def main():
    # Configura o nível de log
    set_global_verbose_eegdata("WARNING")

    # Carrega o dataset Physionet Motor Imagery
    phy = PhysionetMI.loading(verbose="WARNING")
    print("\n\n-------------------------------------------------------- Informações Iniciais e explicação do dataset --------------------------------------------------------")
    print("\nFormato dos dados:", phy.data.shape)  # Mostra o formato dos dados
    print("180 trials (execução de uma tarefa), 1 conjunto por trial, cada trial contém sinais gravados por 64 canais (eletrodos) e para cada canal, há 640 pontos no tempo.")
    # Mostra os labels das tarefas motoras
    print("\nLabels disponíveis: \n", phy.labels)
    print("\nLabel 1: Movimentação ou imaginação de movimento da mão esquerda.")
    print("Label 2: Movimentação ou imaginação de movimento da mão direita.")
    print("Label 3: Movimentação ou imaginação de movimento dos pés.")
    print("Label 4: Movimentação ou imaginação de movimento da língua.")

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

    # Avalia a acurácia antes do alinhamento
    print("\n-------------------------------------------------------- Resultados antes do alinhamento Euclidiano --------------------------------------------------------")
    evaluate_accuracy(original_df_values, phy.labels)
    print("\nGerando gráficos para os dados pré alinhamento")
    # Gráfico por classe antes do alinhamento
    plot_classes(phy, original_df_values)
    # Histograma antes do alinhamento
    plot_sorted_histogram(original_df_values)

    # Avalia a acurácia depois do alinhamento
    print("\n-------------------------------------------------------- Resultados depois do alinhamento Euclidiano --------------------------------------------------------")
    print("\nAcurácia nos labels 1 e 2: x%")
    print("\nGerando gráficos para os dados alinhados")

    # Aplica o alinhamento euclidiano nos dados
    aligned_data = euclidean_alignment(X)

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

    # Gráfico por classe após o alinhamento
    plot_classes(phy, aligned_df_values)
    plot_sorted_histogram(aligned_df_values)  # Histograma após o alinhamento

    # Compara os valores antes e depois do alinhamento
    compare_before_after(aligned_df_values, original_df_values)

    # Gera o gráfico de comparação LDA vs QDA
    mask = (phy.labels == 1) | (phy.labels == 2)  # Filtra labels 1 e 2
    # Usa as duas primeiras dimensões para visualização
    X_reduced = original_df_values[mask][:, :2]
    y_reduced = phy.labels[mask]
    plot_lda_qda_comparison(X_reduced, y_reduced)


if __name__ == "__main__":
    main()
