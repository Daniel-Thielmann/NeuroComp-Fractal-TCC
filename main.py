import os
import numpy as np
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon
import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Create results directory structure
os.makedirs('results/Higuchi', exist_ok=True)
os.makedirs('results/LogPower', exist_ok=True)


class HiguchiFractal:
    def __init__(self, kmax=10):
        self.kmax = kmax

    def _higuchi_fd(self, time_series):
        """Calculate Higuchi Fractal Dimension for a time series"""
        N = len(time_series)
        L = []

        for k in range(1, self.kmax + 1):
            Lk = []
            for m in range(k):
                # Create segments
                ix = np.arange(m, N, k)
                if len(ix) < 2:
                    continue

                # Calculate length
                diff = np.diff(time_series[ix])
                Lkm = np.sum(np.abs(diff)) * (N - 1) / (len(ix) * k)
                Lk.append(Lkm)

            if Lk:  # Only append if we have values
                L.append(np.log(np.mean(Lk)))

        if len(L) < 2:
            return 0.0

        # Fit line to get fractal dimension
        lnk = np.log(1.0 / np.arange(1, len(L)+1))
        return np.polyfit(lnk, L, 1)[0]

    def extract(self, data):
        """Extract features from EEG data (trials x channels x samples)"""
        X = []
        for trial in data:
            # Calculate HFD for each channel
            trial_features = []
            for channel in trial:
                # Apply bandpass filter (0.5-40 Hz) first
                filtered = self._bandpass_filter(channel)
                hfd = self._higuchi_fd(filtered)
                trial_features.append(hfd)
            X.append(trial_features)
        return np.array(X)

    def _bandpass_filter(self, signal, low=0.5, high=40, sfreq=512):
        """Simple bandpass filter"""
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * sfreq
        low = low / nyq
        high = high / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)


class LogPower:
    def __init__(self, freq_bands=None):
        self.freq_bands = freq_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }

    def extract(self, data):
        """Extract log power features from EEG data"""
        X = []
        for trial in data:
            trial_features = []
            for channel in trial:
                # Calculate power in each frequency band
                band_powers = []
                for band, (low, high) in self.freq_bands.items():
                    power = self._bandpower(channel, low, high)
                    # Add small constant to avoid log(0)
                    band_powers.append(np.log(power + 1e-6))
                trial_features.extend(band_powers)
            X.append(trial_features)
        return np.array(X)

    def _bandpower(self, signal, low, high, sfreq=512):
        """Compute power in specific frequency band"""
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=sfreq)
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.mean(psd[idx])


class EEGProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.channel_names = ['F3', 'FC3', 'C3', 'CP3', 'P3',
                              'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']

    def load_subject_data(self, subject_id, data_type='T'):
        """Load data for a single subject"""
        filename = f"parsed_P{subject_id:02d}{data_type}.mat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            logging.error(f"File not found: {filename}")
            return None, None

        try:
            mat = scipy.io.loadmat(filepath)

            # Get data and labels
            data = mat['RawEEGData']  # trials x channels x samples
            labels = mat['Labels'].flatten()

            logging.info(f"Loaded {filename} - Shape: {data.shape}")
            return data, labels

        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            return None, None

    def run_experiment(self, method_name, extractor, subject_ids):
        """Run complete experiment for all subjects"""
        all_results = []
        accuracies = []

        for subject_id in tqdm(subject_ids, desc=f"Running {method_name}"):
            try:
                data, labels = self.load_subject_data(subject_id, 'T')
                if data is None or labels is None:
                    raise ValueError("Invalid data")

                # Create pipeline with better parameters
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LDA(solver='lsqr', shrinkage='auto'))
                ])

                # Prepare for cross-validation
                X = extractor.extract(data)
                skf = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42)
                fold_results = []

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]

                    # Train and predict
                    pipeline.fit(X_train, y_train)
                    probas = pipeline.predict_proba(X_test)

                    # Store fold results
                    for i, (true_label, pred_proba) in enumerate(zip(y_test, probas)):
                        fold_results.append({
                            'subject_id': subject_id,
                            'fold': fold + 1,
                            'true_label': true_label,
                            'left_prob': pred_proba[0],
                            'right_prob': pred_proba[1]
                        })

                # Save subject results
                subject_df = pd.DataFrame(fold_results)
                subject_file = f'results/{method_name}/P{subject_id:02d}.csv'
                subject_df.to_csv(subject_file, index=False)

                # Calculate subject accuracy
                y_pred = pipeline.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                accuracies.append(accuracy)
                logging.info(
                    f"{method_name} - P{subject_id:02d}: Accuracy = {accuracy:.4f}")

                # Add to all results
                all_results.append(subject_df)

            except Exception as e:
                logging.error(
                    f"Error processing subject {subject_id:02d}: {str(e)}")
                accuracies.append(np.nan)

        # Save combined results
        if all_results:
            final_df = pd.concat(all_results)
            final_file = f'results/{method_name}_final.csv'
            final_df.to_csv(final_file, index=False)
            logging.info(f"Saved final results to {final_file}")

        return accuracies


def main():
    DATA_DIR = "data/wcci2020/"

    # Verify data directory
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return

    processor = EEGProcessor(DATA_DIR)
    subject_ids = range(1, 11)  # P01 to P10

    # Initialize feature extractors with improved parameters
    # Increased kmax for better HFD estimation
    higuchi = HiguchiFractal(kmax=15)
    logpower = LogPower()  # Using default frequency bands

    # Run experiments for both methods
    higuchi_acc = processor.run_experiment("Higuchi", higuchi, subject_ids)
    logpower_acc = processor.run_experiment("LogPower", logpower, subject_ids)

    # Create accuracy report
    accuracy_df = pd.DataFrame({
        'Subject': [f'P{i:02d}' for i in subject_ids],
        'Higuchi': higuchi_acc,
        'LogPower': logpower_acc
    })

    # Calculate mean accuracies
    mean_higuchi = np.nanmean(higuchi_acc)
    mean_logpower = np.nanmean(logpower_acc)

    print("\n=== Accuracy Report ===")
    print(accuracy_df)
    print(f"\nHiguchi Mean Accuracy: {mean_higuchi:.4f}")
    print(f"LogPower Mean Accuracy: {mean_logpower:.4f}")

    # Wilcoxon signed-rank test
    valid_pairs = [(h, l) for h, l in zip(higuchi_acc, logpower_acc)
                   if not np.isnan(h) and not np.isnan(l)]

    if len(valid_pairs) >= 2:
        hig_valid, log_valid = zip(*valid_pairs)
        stat, p = wilcoxon(hig_valid, log_valid)
        print("\n=== Wilcoxon Test Results ===")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p:.4f}")

        if p < 0.05:
            print("Conclusion: Significant difference between methods (p < 0.05)")
        else:
            print("Conclusion: No significant difference between methods")


if __name__ == "__main__":
    main()
