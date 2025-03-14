import numpy as np


class KatzFractal:
    """
    Classe que aplica a Dimensão Fractal de Katz a um conjunto de sinais EEG.
    Implementa um método fit_transform para compatibilidade com o framework bciflow.
    """

    def __init__(self):
        pass

    def katz_fd(self, signal):
        """
        Calcula a Dimensão Fractal de Katz para um sinal EEG.

        Parâmetros:
        - signal: Um array 1D representando o sinal EEG.

        Retorno:
        - Valor da Dimensão Fractal de Katz para o sinal.
        """
        L = np.sum(np.abs(np.diff(signal)))  # Comprimento total do sinal
        # Distância máxima do primeiro ponto ao mais distante
        d = np.max(np.abs(signal - signal[0]))
        N = len(signal)

        return np.log10(N) / (np.log10(N) + np.log10(L/d))

    def fit_transform(self, eegdata: dict, **kwargs):
        """
        Aplica a Dimensão Fractal de Katz aos sinais EEG e atualiza o dicionário de dados.

        Parâmetros:
        - eegdata: Dicionário contendo os dados EEG no formato 'X'.

        Retorno:
        - eegdata atualizado com as novas features baseadas na DF de Katz.
        """
        X = eegdata['X'].copy()
        X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

        # Aplica a função de Katz para cada sinal EEG
        X_ = np.array([self.katz_fd(X[i]) for i in range(X.shape[0])])

        # Reformatar para o formato original
        shape = eegdata['X'].shape
        X_ = X_.reshape((shape[0], np.prod(shape[1:-1])))

        eegdata['X'] = X_
        return eegdata
