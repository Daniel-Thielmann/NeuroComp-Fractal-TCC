import numpy as np
from sklearn.preprocessing import StandardScaler


class HiguchiFractal:
    def __init__(self, kmax=10):
        self.kmax = kmax

    def _higuchi_fd(self, time_series):
        L = []
        x = time_series
        N = len(x)
        kmax = self.kmax

        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
                length = np.sum(
                    np.abs(x[m + idxs * k] - x[m + (idxs - 1) * k]))
                norm = (N - 1) / (len(idxs) * k)
                Lm = (length * norm)
                Lk.append(Lm)
            L.append(np.mean(Lk))

        lnL = np.log(L)
        lnk = np.log(1.0 / np.arange(1, kmax + 1))
        return np.polyfit(lnk, lnL, 1)[0]

    def extract(self, eeg):
        X = []
        y = eeg.events["labels"]

        for trial in eeg.data:
            trial_features = [self._higuchi_fd(channel) for channel in trial]
            X.append(trial_features)

        return StandardScaler().fit_transform(X), y
