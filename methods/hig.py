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
        N = len(signal)
        k_vals = np.arange(1, self.kmax + 1)
        L_vals = np.zeros_like(k_vals, dtype=np.float64)

        for idx, k in enumerate(k_vals):
            Lk = np.zeros(k)
            for m in range(k):
                indices = np.arange(m, N, k)
                diff = np.abs(np.diff(signal[indices]))
                if len(indices) > 1:
                    Lmk = np.sum(diff) / (len(indices) * k)
                    Lk[m] = Lmk
            L_vals[idx] = np.mean(Lk) if len(Lk) > 0 else 0

        if np.any(L_vals > 0):
            return -np.polyfit(np.log(k_vals + 1e-10), np.log(L_vals + 1e-10), 1)[0]
        return 0

    def fit_transform(self, eegdata: dict, **kwargs):
        """
        Método usado pelo `bciflow` para aplicar a transformação aos dados EEG.
        Esse método apenas chama `transform()`.
        """
        return self.transform(eegdata)

    def transform(self, eegdata: dict, **kwargs):
        """
        Aplica a DF de Higuchi a cada canal do EEG.
        """
        start_time = time.time()
        print(
            f"[Higuchi] Calculando DF para {eegdata['X'].shape[0]} trials...")

        X = eegdata['X'].copy()
        X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
        X_ = np.array([self.higuchi_fd(X[i]) for i in range(X.shape[0])])

        shape = eegdata['X'].shape
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))
        eegdata['X'] = X_

        elapsed_time = time.time() - start_time
        print(
            f"[Higuchi] Concluído em {elapsed_time:.2f}s | Exemplo: {X_[:1].round(4)}")

        return eegdata
