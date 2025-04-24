import numpy as np


class LogPower:
    def __init__(self, epsilon=1e-6, flatting=True):
        self.epsilon = epsilon
        self.flatting = flatting

    def fit(self, X, y=None):
        return self

    def transform(self, eegdata):
        """
        Aplica o cálculo da potência logarítmica aos dados EEG do dicionário.
        Espera que eegdata seja um dicionário no formato {'X': np.ndarray}.
        """
        try:
            if not isinstance(eegdata, dict) or 'X' not in eegdata:
                raise ValueError(
                    "Esperado dicionário com chave 'X' contendo os dados EEG.")

            X = eegdata['X']  # Extrai os dados EEG do dicionário

            # Cálculo da potência logarítmica
            power = np.mean(X ** 2, axis=-1)
            result = np.log10(power + self.epsilon)

            # Ajuste de forma (opcional)
            if self.flatting:
                result = result.reshape((result.shape[0], -1))

            # Substitui os dados originais pelo resultado
            eegdata['X'] = result
            return eegdata

        except Exception as e:
            raise ValueError(
                f"Erro ao transformar os dados com LogPower: {str(e)}")

    def fit_transform(self, X, y=None):
        return self.transform(X)
