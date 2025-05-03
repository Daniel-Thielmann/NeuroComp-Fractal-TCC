import os
import numpy as np
import scipy.io
import pandas as pd
import logging
from tqdm import tqdm
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.stats import wilcoxon
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
os.makedirs('results/Higuchi', exist_ok=True)
os.makedirs('results/LogPower', exist_ok=True)


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


class LogPowerEnhanced:
    def __init__(self, freq_bands=None, sfreq=512):
        self.freq_bands = freq_bands or [
            ('delta', 0.5, 4),
            ('theta', 4, 8),
            ('alpha', 8, 13),
            ('beta', 13, 30),
            ('gamma', 30, 40)
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


class EEGProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_subject_data(self, subject_id, data_type='T'):
        filename = f"parsed_P{subject_id:02d}{data_type}.mat"
        filepath = os.path.join(self.data_dir, filename)
        try:
            mat = scipy.io.loadmat(filepath)
            return mat['RawEEGData'], mat['Labels'].flatten()
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            return None, None

    def run_experiment(self, method_name, extractor, subject_ids):
        accuracies = []
        for subject_id in tqdm(subject_ids, desc=f"Running {method_name}"):
            data, labels = self.load_subject_data(subject_id)
            if data is None:
                accuracies.append(np.nan)
                continue

            X = extractor.extract(data)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_acc = []

            for train_idx, test_idx in skf.split(X, labels):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                scaler = StandardScaler()
                pca = PCA(n_components=0.95)

                X_train = scaler.fit_transform(X_train)
                X_train = pca.fit_transform(X_train)

                X_test = scaler.transform(X_test)
                X_test = pca.transform(X_test)

                n_features_post_pca = X_train.shape[1]
                k = min(30, n_features_post_pca)
                if k < n_features_post_pca:
                    selector = SelectKBest(f_classif, k=k)
                    X_train = selector.fit_transform(X_train, y_train)
                    X_test = selector.transform(X_test)

                clf = LDA(solver='lsqr', shrinkage='auto')
                clf.fit(X_train, y_train)
                fold_acc.append(clf.score(X_test, y_test))

            acc = np.mean(fold_acc)
            accuracies.append(acc)
            logging.info(
                f"{method_name} - P{subject_id:02d}: Accuracy = {acc:.4f}")

        return accuracies


def main():
    DATA_DIR = "data/wcci2020/"
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return

    processor = EEGProcessor(DATA_DIR)
    subject_ids = range(1, 11)

    higuchi = HiguchiFractalEvolution(kmax=10)
    logpower = LogPowerEnhanced()

    higuchi_acc = processor.run_experiment("HFE", higuchi, subject_ids)
    logpower_acc = processor.run_experiment("LogPower", logpower, subject_ids)

    report = pd.DataFrame({
        'Subject': [f'P{i:02d}' for i in subject_ids],
        'HFE': higuchi_acc,
        'LogPower': logpower_acc
    })

    print("\n=== Accuracy Report ===")
    print(report)
    print(f"\nHFE Mean: {np.nanmean(higuchi_acc):.4f}")
    print(f"LogPower Mean: {np.nanmean(logpower_acc):.4f}")

    valid_pairs = [(h, l) for h, l in zip(higuchi_acc, logpower_acc)
                   if not np.isnan(h) and not np.isnan(l)]
    if len(valid_pairs) >= 2:
        hig_valid, log_valid = zip(*valid_pairs)
        stat, p = wilcoxon(hig_valid, log_valid)
        print("\n=== Wilcoxon Test ===")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p:.4f}")
        if p < 0.05:
            print("Conclusion: Significant difference between methods (p < 0.05)")
            print("HFE is significantly better!" if np.mean(hig_valid) >
                  np.mean(log_valid) else "LogPower is significantly better!")
        else:
            print("Conclusion: No significant difference between methods")


if __name__ == "__main__":
    main()
