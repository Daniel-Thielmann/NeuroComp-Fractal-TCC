import numpy as np
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from methods.features.fractal import HiguchiFractalEvolution

def run_csp_fractal(subject_id, data_path="dataset/wcci2020/"):
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"].squeeze(1)
    y = np.array(dataset["y"]) + 1

    X_band = np.expand_dims(X, axis=1)
    transformer = csp()
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})["X"][:, 0]

    hfd = HiguchiFractalEvolution(kmax=100)
    fd_features = []
    for trial in X_csp:
        trial_feat = []
        for comp in trial:
            comp = comp - np.mean(comp)
            slope, mean_lk, std_lk = hfd._calculate_enhanced_hfd(comp)
            trial_feat.extend([slope, mean_lk, std_lk])
        fd_features.append(trial_feat)
    fd_features = np.array(fd_features)

    fd_features = StandardScaler().fit_transform(fd_features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(fd_features, y)):
        X_train, X_test = fd_features[train_idx], fd_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LDA()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)

        for i, idx in enumerate(test_idx):
            rows.append({
                "subject_id": subject_id,
                "fold": fold_idx,
                "true_label": y_test[i],
                "left_prob": probs[i][0],
                "right_prob": probs[i][1],
            })

    return rows
