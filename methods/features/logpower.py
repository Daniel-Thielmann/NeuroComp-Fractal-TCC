import numpy as np
from scipy.signal import welch


class LogPower:
    def __init__(self, freq_bands=None, sfreq=512):
        self.freq_bands = freq_bands or [
            ("delta", 0.5, 4),
            ("theta", 4, 8),
            ("alpha", 8, 13),
            ("beta", 13, 30),
            ("gamma", 30, 40),
        ]
        self.sfreq = sfreq

    def extract(self, data):
        n_trials, n_channels, _ = data.shape
        n_bands = len(self.freq_bands)
        X = np.zeros((n_trials, n_channels * n_bands))

        for i in range(n_trials):
            for j in range(n_channels):
                freqs, psd = welch(data[i, j, :], fs=self.sfreq, nperseg=128)
                for k, (_, low, high) in enumerate(self.freq_bands):
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        power = np.mean(psd[mask])
                        X[i, j * n_bands + k] = np.log(power + 1e-12)
        return X
