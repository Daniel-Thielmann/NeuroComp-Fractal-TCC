import numpy as np
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from bciflow.modules.tf.filterbank import filterbank
from methods.features.fractal import HiguchiFractalEvolution
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_fbcsp_fractal(subject_id, data_path="dataset/wcci2020/"):
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"]
    y = np.array(dataset["y"]) + 1

    bands = [(8, 12), (13, 30), (8, 13)]  # mu, beta, alpha
    eeg_filtered = filterbank({"X": X, "sfreq": 512}, kind_bp="chebyshevII")
    X_filtered = eeg_filtered["X"]

    transformer = csp()
    transformer.fit({"X": X_filtered, "y": y})
    X_csp = transformer.transform({"X": X_filtered})["X"]

    hfd = HiguchiFractalEvolution(kmax=80)
    features = []
    for trial in X_csp:
        trial_feat = []
        for band in trial:
            for comp in band:
                slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(comp)
                trial_feat.extend([slope, mean_lk, std_lk])
        features.append(trial_feat)
    features = np.array(features)

    features = StandardScaler().fit_transform(features)
    features = PCA(n_components=10).fit_transform(features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LDA()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)

        for i, idx in enumerate(test_idx):
            rows.append(
                {
                    "subject_id": subject_id,
                    "fold": fold_idx,
                    "true_label": y_test[i],
                    "left_prob": probs[i][0],
                    "right_prob": probs[i][1],
                }
            )

    return rows
