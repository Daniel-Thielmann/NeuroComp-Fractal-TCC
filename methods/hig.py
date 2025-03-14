import numpy as np
import time


class HiguchiFractal:
    """
    Classe que implementa a Dimensão Fractal de Higuchi otimizada para sinais EEG.
    Compatível com `bciflow`.
    """

    def __init__(self, kmax=12):
        """
        Inicializa a classe HiguchiFractal.

        Parâmetros:
        - kmax: Número máximo de divisões do sinal para calcular a DF de Higuchi.
        """
        self.kmax = kmax

    def higuchi_fd(self, signal):
        """
        Calcula a Dimensão Fractal de Higuchi para um único sinal EEG.

        Parâmetros:
        - signal: Um array 1D representando o sinal EEG de um eletrodo.

        Retorno:
        - O valor da Dimensão Fractal de Higuchi para aquele sinal.
        """
        N = len(signal)  # Número total de amostras no sinal
        # Lista de valores de "k" que dividem o sinal
        k_vals = np.arange(1, self.kmax + 1)
        # Armazena os comprimentos médios para cada "k"
        L_vals = np.zeros_like(k_vals, dtype=np.float64)

        for idx, k in enumerate(k_vals):  # Para cada valor de k
            # Lista para armazenar os comprimentos normalizados
            Lk = np.zeros(k)
            for m in range(k):  # Para cada deslocamento "m" dentro de "k"
                # Criamos uma sequência de índices espaçados por "k"
                indices = np.arange(m, N, k)
                # Calculamos a diferença entre valores consecutivos
                diff = np.abs(np.diff(signal[indices]))
                if len(indices) > 1:
                    # Comprimento médio normalizado
                    Lmk = np.sum(diff) / (len(indices) * k)
                    Lk[m] = Lmk  # Armazena o comprimento médio para esse "m"

            # Calcula o valor médio de L(k)
            L_vals[idx] = np.mean(Lk) if len(Lk) > 0 else 0

        if np.any(L_vals > 0):  # Evita erro de log(0)
            return -np.polyfit(np.log(k_vals + 1e-10), np.log(L_vals + 1e-10), 1)[0]
        return 0  # Retorna 0 para evitar NaN

    def fit_transform(self, eegdata: dict, **kwargs):
        """
        Método usado pelo `bciflow` para aplicar a transformação aos dados EEG.
        Esse método apenas chama `transform()`.

        Parâmetros:
        - eegdata: Dicionário contendo os dados EEG no formato {'X': matriz de sinais}.

        Retorno:
        - O mesmo dicionário `eegdata`, mas com os valores substituídos pela DF de Higuchi.
        """
        return self.transform(eegdata)

    def transform(self, eegdata: dict, **kwargs):
        """
        Método que aplica a DF de Higuchi a cada canal do EEG.

        Parâmetros:
        - eegdata: Dicionário contendo os dados EEG no formato {'X': matriz de sinais}.

        Retorno:
        - O mesmo dicionário `eegdata`, mas com os valores da DF de Higuchi para cada eletrodo.
        """
        print("Iniciando cálculo da Dimensão Fractal de Higuchi...")
        start_time = time.time()

        X = eegdata['X'].copy()  # Copia os dados do EEG
        # Reorganiza os dados para serem processados
        X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

        # Aplica a função de Higuchi para cada eletrodo e salva os valores da DF.
        X_ = np.array([self.higuchi_fd(X[i]) for i in range(X.shape[0])])

        # Exibe algumas amostras de saída para depuração
        print(f"DF de Higuchi calculada para {X.shape[0]} amostras.")
        print(f"Exemplo de valores de DF: {X_[:5]}")

        # Retorna os dados para o formato original
        shape = eegdata['X'].shape
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))

        eegdata['X'] = X_  # Substitui os dados originais pelos valores da DF

        elapsed_time = time.time() - start_time
        print(f"Cálculo da DF concluído em {elapsed_time:.2f} segundos.")

        return eegdata
