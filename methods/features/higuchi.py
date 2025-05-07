import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert

class HiguchiFractalEvolution:
    def __init__(self, kmax=10, bands=None, sfreq=512):
        self.kmax = kmax
        self.bands = bands or [
            ('delta', 0.5, 4),
            ('theta', 4, 8),
            ('alpha', 8, 13),
            ('beta', 13, 30),
            ('gamma', 30, 40)
        ]
        self.sfreq = sfreq
        self.filter_bank = self._create_filter_bank()

    def _create_filter_bank(self):
        filter_bank = {}
        nyq = 0.5 * self.sfreq
        for name, low, high in self.bands:
            b, a = butter(4, [low / nyq, high / nyq], btype='band')
            filter_bank[name] = (b, a)
        return filter_bank

    def _calculate_enhanced_hfd(self, signal):
        n = len(signal)
        if n < 10:
            return 0.0, np.zeros(self.kmax)

        scales = np.unique(np.logspace(0, np.log10(
            min(self.kmax, n // 2)), num=10, dtype=int))
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
            return 0.0, lk

        hfd = np.polyfit(np.log(1.0 / scales[valid]), lk[valid], 1)[0]
        return hfd, lk

    def _extract_time_domain_features(self, signal):
        analytic_signal = hilbert(signal)
        amplitude = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))

        features = [
            np.mean(amplitude),
            np.std(amplitude),
            np.mean(np.diff(phase)),
            np.std(np.diff(phase)),
            len(np.where(np.diff(np.sign(signal)))[0]) / len(signal),
            np.max(amplitude) - np.min(amplitude)
        ]
        return features

    def _calculate_band_features(self, signal, band_name, low, high):
        b, a = self.filter_bank[band_name]
        filtered = filtfilt(b, a, signal)
        hfd, hfd_profile = self._calculate_enhanced_hfd(filtered)

        freqs, psd = welch(filtered, fs=self.sfreq, nperseg=128)
        mask = (freqs >= low) & (freqs <= high)

        if np.any(mask):
            spectral_features = [
                np.log(np.mean(psd[mask]) + 1e-12),
                -np.sum(psd[mask] * np.log(psd[mask] + 1e-12)),
                freqs[mask][np.argmax(psd[mask])]
            ]
        else:
            spectral_features = [0.0, 0.0, 0.0]

        time_features = self._extract_time_domain_features(filtered)
        return [hfd] + spectral_features + time_features + list(hfd_profile)

    def extract(self, data):
        n_trials, n_channels, _ = data.shape
        hfd_profile_len = len(np.unique(np.logspace(
            0, np.log10(self.kmax), num=10, dtype=int)))
        features_per_band = 1 + 3 + 6 + hfd_profile_len
        X = np.zeros(
            (n_trials, n_channels * len(self.bands) * features_per_band))

        for i in range(n_trials):
            for j in range(n_channels):
                for k, (band_name, low, high) in enumerate(self.bands):
                    start = (j * len(self.bands) + k) * features_per_band
                    end = start + features_per_band
                    features = self._calculate_band_features(
                        data[i, j, :], band_name, low, high)
                    X[i, start:end] = features[:features_per_band]
        return X

def higuchi_fd(signal, kmax=10):
    L = []
    x = np.asarray(signal)
    N = x.size
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
            Lmk /= k * ((N - 1) / k)
            Lk += Lmk
        L.append(np.log(Lk / k))
    return -np.polyfit(np.log(range(1, kmax + 1)), L, 1)[0]
