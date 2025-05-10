import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert


class HiguchiFractalEvolution:
    def __init__(self, kmax=80, bands=None, sfreq=512):
        self.kmax = kmax
        self.bands = bands or [("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]
        self.sfreq = sfreq
        self.filter_bank = self._create_filter_bank()

    def _create_filter_bank(self):
        filter_bank = {}
        nyq = 0.5 * self.sfreq
        for name, low, high in self.bands:
            b, a = butter(4, [low / nyq, high / nyq], btype="band")
            filter_bank[name] = (b, a)
        return filter_bank

    def _calculate_enhanced_hfd(self, signal):
        n = len(signal)
        if n < 10:
            return 0.0, 0.0, 0.0

        scales = np.unique(
            np.logspace(0, np.log10(min(self.kmax, n // 2)), num=10, dtype=int)
        )
        lk = np.zeros(len(scales))
        diff = np.abs(np.diff(signal))

        for i, k in enumerate(scales):
            sum_l = 0.0
            count = 0
            for m in range(k):
                ix = np.arange(m, n, k)
                if len(ix) > 1:
                    sum_l += np.sum(diff[ix[:-1]]) * (n - 1) / (len(ix) * k)
                    count += 1
            lk[i] = np.log(sum_l / count) if count > 0 else 0.0

        valid = (lk != 0) & ~np.isinf(lk)
        if np.sum(valid) < 2:
            return 0.0, 0.0, 0.0

        slope = np.polyfit(np.log(1.0 / scales[valid]), lk[valid], 1)[0]
        return slope, np.mean(lk[valid]), np.std(lk[valid])

    def extract(self, data):
        n_trials, n_channels, _ = data.shape
        n_features_per_band = 3  # slope, mean, std
        total_features = n_channels * len(self.bands) * n_features_per_band
        X = np.zeros((n_trials, total_features))

        for i in range(n_trials):
            for j in range(n_channels):
                for k, (band_name, low, high) in enumerate(self.bands):
                    b, a = self.filter_bank[band_name]
                    filtered = filtfilt(b, a, data[i, j, :])
                    slope, mean_lk, std_lk = self._calculate_enhanced_hfd(filtered)
                    idx = (j * len(self.bands) + k) * n_features_per_band
                    X[i, idx : idx + 3] = [slope, mean_lk, std_lk]
        return X
