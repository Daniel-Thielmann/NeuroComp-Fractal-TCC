import numpy as np
from bciflow.datasets import cbcic
from bciflow.modules.sf.csp import csp
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from methods.features.higuchi import higuchi_fd

def run_csp_fractal(subject_id, data_path="dataset/wcci2020/"):
    dataset = cbcic(subject=subject_id, path=data_path)
    X = dataset["X"].squeeze(1)  # [n_trials, channels, samples]
    y = np.array(dataset["y"])

    X_band = np.expand_dims(X, axis=1)  # [n_trials, 1, channels, samples]
    transformer = csp()
    transformer.fit({"X": X_band, "y": y})
    X_csp = transformer.transform({"X": X_band})[:, 0]  # [n_trials, components, samples]

    fd_features = []
    for trial in X_csp:
        trial_fd = [higuchi_fd(component) for component in trial]
        fd_features.append(trial_fd)
    fd_features = np.array(fd_features)

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
                'subject_id': subject_id,
                'fold': fold_idx,
                'true_label': y_test[i],
                'left_prob': probs[i][0],
                'right_prob': probs[i][1],
            })

    return rows
